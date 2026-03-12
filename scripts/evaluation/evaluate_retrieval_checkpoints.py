#!/usr/bin/env python
"""Evaluate embedding checkpoints on the FamilyOS retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "data" / "familyos" / "benchmarks" / "retrieval_golden_v1"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel


SELECTION_WEIGHTS: dict[str, float] = {
    "recall_at_1": 0.40,
    "recall_at_5": 0.20,
    "ndcg_at_10": 0.20,
    "mrr": 0.10,
    "triplet_accuracy": 0.10,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def reciprocal_rank(rank: int) -> float:
    """Return reciprocal rank for a 1-indexed rank."""
    return 0.0 if rank <= 0 else 1.0 / float(rank)


def ndcg_at_k(rank: int, k: int) -> float:
    """Return nDCG@k for a single relevant document at the given rank."""
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(float(rank) + 1.0)


def summarize_rank_metrics(ranks: list[int], k_values: tuple[int, ...] = (1, 5, 10)) -> dict[str, Any]:
    """Summarize retrieval metrics for a list of 1-indexed ranks."""
    if not ranks:
        return {
            "n_queries": 0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0,
            "ndcg@10": 0.0,
            "median_rank": None,
            "mean_rank": None,
            "worst_rank": None,
        }

    sorted_ranks = sorted(ranks)
    n_queries = len(ranks)
    summary: dict[str, Any] = {
        "n_queries": n_queries,
        "mrr": sum(reciprocal_rank(rank) for rank in ranks) / n_queries,
        "ndcg@10": sum(ndcg_at_k(rank, 10) for rank in ranks) / n_queries,
        "median_rank": sorted_ranks[n_queries // 2] if n_queries % 2 == 1 else (sorted_ranks[(n_queries // 2) - 1] + sorted_ranks[n_queries // 2]) / 2.0,
        "mean_rank": sum(ranks) / n_queries,
        "worst_rank": max(ranks),
    }

    for k in k_values:
        summary[f"recall@{k}"] = sum(rank <= k for rank in ranks) / n_queries

    return summary


def compute_selection_score(retrieval: dict[str, Any], triplets: dict[str, Any]) -> dict[str, Any]:
    """Compute a retrieval-first combined selection score."""
    metric_values = {
        "recall_at_1": float(retrieval.get("recall@1", 0.0)),
        "recall_at_5": float(retrieval.get("recall@5", 0.0)),
        "ndcg_at_10": float(retrieval.get("ndcg@10", 0.0)),
        "mrr": float(retrieval.get("mrr", 0.0)),
        "triplet_accuracy": float(triplets.get("accuracy", 0.0)),
    }
    weighted_components = {
        name: metric_values[name] * weight for name, weight in SELECTION_WEIGHTS.items()
    }
    return {
        "score": sum(weighted_components.values()),
        "weights": SELECTION_WEIGHTS,
        "metrics": metric_values,
        "components": weighted_components,
    }


def cap_records_balanced(records: list[dict[str, Any]], max_records: int) -> list[dict[str, Any]]:
    """Cap records while preserving slice coverage as evenly as possible."""
    if max_records <= 0 or len(records) <= max_records:
        return records

    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(record["slice"], []).append(record)

    slice_names = sorted(buckets.keys())
    per_slice = max(1, max_records // max(1, len(slice_names)))
    capped: list[dict[str, Any]] = []

    for slice_name in slice_names:
        capped.extend(buckets[slice_name][:per_slice])

    if len(capped) < max_records:
        leftovers: list[dict[str, Any]] = []
        for slice_name in slice_names:
            leftovers.extend(buckets[slice_name][per_slice:])
        capped.extend(leftovers[: max_records - len(capped)])

    return capped[:max_records]


def load_tokenizer(checkpoint_path: Path) -> Any:
    """Load a tokenizer from checkpoint with a local fast-tokenizer fallback."""
    try:
        return AutoTokenizer.from_pretrained(str(checkpoint_path), local_files_only=True)
    except ValueError as exc:
        tokenizer_config_path = checkpoint_path / "tokenizer_config.json"
        tokenizer_json_path = checkpoint_path / "tokenizer.json"
        if not tokenizer_config_path.exists() or not tokenizer_json_path.exists():
            raise exc

        with open(tokenizer_config_path, encoding="utf-8") as handle:
            tokenizer_config = json.load(handle)

        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json_path),
            cls_token=tokenizer_config.get("cls_token", "[CLS]"),
            sep_token=tokenizer_config.get("sep_token", "[SEP]"),
            pad_token=tokenizer_config.get("pad_token", "[PAD]"),
            mask_token=tokenizer_config.get("mask_token", "[MASK]"),
            unk_token=tokenizer_config.get("unk_token", "[UNK]"),
            model_max_length=int(tokenizer_config.get("model_max_length", 512)),
            truncation_side=tokenizer_config.get("truncation_side", "right"),
            clean_up_tokenization_spaces=bool(
                tokenizer_config.get("clean_up_tokenization_spaces", True)
            ),
        )


def head_supports_mode_routing(head: torch.nn.Module) -> bool:
    """Return whether the embedding head supports query/document routing."""
    return getattr(head, "pooling", None) == "agreement_gated_v2"


def forward_embedding_head(
    head: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """Forward the embedding head with backward-compatible mode routing."""
    if head_supports_mode_routing(head):
        output = head(hidden_states, attention_mask, mode=mode, return_aux=False)
        if isinstance(output, dict):
            return output["embedding"]
        return output
    return head(hidden_states, attention_mask)


def encode_texts(
    model: ModernBertMultiTaskModel,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    mode: str,
    max_length: int,
) -> torch.Tensor:
    """Encode texts into normalized embedding vectors."""
    outputs: list[torch.Tensor] = []
    head = model.heads["embedding"]
    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            encoder_output = model.encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            )
            hidden_states = encoder_output.last_hidden_state
            embeddings = forward_embedding_head(
                head=head,
                hidden_states=hidden_states,
                attention_mask=encoded["attention_mask"],
                mode=mode,
            )
            outputs.append(F.normalize(embeddings, p=2, dim=-1).cpu())

    return torch.cat(outputs, dim=0) if outputs else torch.zeros((0, 0), dtype=torch.float32)


def evaluate_pair_records(
    model: ModernBertMultiTaskModel,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int,
    audit_top_k: int,
    max_failure_examples: int,
    persist_topk_audit: bool,
) -> dict[str, Any]:
    """Evaluate retrieval pairs with a shared candidate pool."""
    if not records:
        return {
            "aggregate": summarize_rank_metrics([]),
            "by_slice": {},
            "by_difficulty": {},
            "candidate_docs": 0,
            "failure_examples": [],
            "query_audit": [],
        }

    documents: list[str] = []
    document_index: dict[str, int] = {}
    gold_indices: list[int] = []
    queries = [record["query"] for record in records]

    for record in records:
        positive = record["positive"]
        if positive not in document_index:
            document_index[positive] = len(documents)
            documents.append(positive)
        gold_indices.append(document_index[positive])

    query_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=queries,
        device=device,
        batch_size=batch_size,
        mode="query",
        max_length=max_length,
    )
    document_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=documents,
        device=device,
        batch_size=batch_size,
        mode="document",
        max_length=max_length,
    )

    similarity = query_embeddings @ document_embeddings.T
    ranks: list[int] = []
    slice_buckets: dict[str, list[int]] = {}
    difficulty_buckets: dict[str, list[int]] = {}
    failure_examples: list[dict[str, Any]] = []
    query_audit: list[dict[str, Any]] = []

    for row_index, gold_index in enumerate(gold_indices):
        order = torch.argsort(similarity[row_index], descending=True)
        rank = int((order == gold_index).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)

        record = records[row_index]
        slice_name = str(record.get("slice", "unknown"))
        difficulty = str(record.get("difficulty", "unknown"))
        slice_buckets.setdefault(slice_name, []).append(row_index)
        difficulty_buckets.setdefault(difficulty, []).append(row_index)

        top_indices = order[: max(1, audit_top_k)].detach().cpu().tolist()
        top_docs = [
            {
                "rank": idx + 1,
                "doc_index": int(doc_index),
                "text": documents[doc_index],
                "score": float(similarity[row_index, doc_index].item()),
                "is_gold": bool(doc_index == gold_index),
            }
            for idx, doc_index in enumerate(top_indices)
        ]
        gold_score = float(similarity[row_index, gold_index].item())
        top1_score = float(similarity[row_index, top_indices[0]].item()) if top_indices else gold_score

        if persist_topk_audit:
            query_audit.append(
                {
                    "benchmark_id": record.get("benchmark_id"),
                    "query": record.get("query"),
                    "slice": slice_name,
                    "difficulty": difficulty,
                    "gold_text": documents[gold_index],
                    "gold_rank": rank,
                    "gold_score": gold_score,
                    "top_k": top_docs,
                }
            )

        if rank > 1:
            failure_examples.append(
                {
                    "benchmark_id": record.get("benchmark_id"),
                    "query": record.get("query"),
                    "slice": slice_name,
                    "difficulty": difficulty,
                    "gold_rank": rank,
                    "gold_text": documents[gold_index],
                    "gold_score": gold_score,
                    "top1_text": documents[top_indices[0]] if top_indices else documents[gold_index],
                    "top1_score": top1_score,
                    "score_gap_vs_top1": gold_score - top1_score,
                    "top_k": top_docs,
                }
            )

    by_slice = {
        slice_name: summarize_rank_metrics([ranks[index] for index in indices])
        for slice_name, indices in sorted(slice_buckets.items())
    }
    by_difficulty = {
        difficulty: summarize_rank_metrics([ranks[index] for index in indices])
        for difficulty, indices in sorted(difficulty_buckets.items())
    }
    failure_examples.sort(key=lambda item: (-int(item["gold_rank"]), item["score_gap_vs_top1"]))

    return {
        "aggregate": summarize_rank_metrics(ranks),
        "candidate_docs": len(documents),
        "by_slice": by_slice,
        "by_difficulty": by_difficulty,
        "failure_examples": failure_examples[: max(0, max_failure_examples)],
        "query_audit": query_audit if persist_topk_audit else [],
    }


def evaluate_triplet_records(
    model: ModernBertMultiTaskModel,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """Evaluate hard-negative triplets by slice and in aggregate."""
    if not records:
        return {"aggregate": {"n": 0, "accuracy": 0.0, "margin": 0.0}, "by_slice": {}}

    anchors = [record["anchor"] for record in records]
    positives = [record["positive"] for record in records]
    negatives = [record["negative"] for record in records]

    anchor_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=anchors,
        device=device,
        batch_size=batch_size,
        mode="document",
        max_length=max_length,
    )
    positive_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=positives,
        device=device,
        batch_size=batch_size,
        mode="document",
        max_length=max_length,
    )
    negative_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=negatives,
        device=device,
        batch_size=batch_size,
        mode="document",
        max_length=max_length,
    )

    pos_sim = F.cosine_similarity(anchor_embeddings, positive_embeddings).numpy().tolist()
    neg_sim = F.cosine_similarity(anchor_embeddings, negative_embeddings).numpy().tolist()

    slice_buckets: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        slice_buckets.setdefault(record["slice"], []).append(index)

    def summarize(indices: list[int]) -> dict[str, Any]:
        margins = [pos_sim[i] - neg_sim[i] for i in indices]
        accuracy = sum(pos_sim[i] > neg_sim[i] for i in indices) / len(indices)
        return {
            "n": len(indices),
            "accuracy": accuracy,
            "margin": sum(margins) / len(margins),
            "pos_sim": sum(pos_sim[i] for i in indices) / len(indices),
            "neg_sim": sum(neg_sim[i] for i in indices) / len(indices),
        }

    by_slice = {slice_name: summarize(indices) for slice_name, indices in sorted(slice_buckets.items())}
    aggregate = summarize(list(range(len(records))))
    return {
        "aggregate": aggregate,
        "by_slice": by_slice,
    }


def evaluate_split(
    model: ModernBertMultiTaskModel,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int,
    audit_top_k: int,
    max_failure_examples: int,
    persist_topk_audit: bool,
) -> dict[str, Any]:
    """Evaluate a full benchmark split."""
    pair_records = [record for record in records if record["task_type"] == "pair"]
    triplet_records = [record for record in records if record["task_type"] == "triplet"]
    retrieval = evaluate_pair_records(
        model,
        tokenizer,
        pair_records,
        device,
        batch_size,
        max_length,
        audit_top_k,
        max_failure_examples,
        persist_topk_audit,
    )
    triplets = evaluate_triplet_records(model, tokenizer, triplet_records, device, batch_size, max_length)
    selection = compute_selection_score(retrieval["aggregate"], triplets["aggregate"])
    return {
        "total_records": len(records),
        "pair_records": len(pair_records),
        "triplet_records": len(triplet_records),
        "retrieval": retrieval,
        "triplets": triplets,
        "selection": selection,
    }


def print_split_summary_table(checkpoint_label: str, checkpoint_result: dict[str, Any]) -> None:
    """Print a compact summary table for all evaluated splits."""
    print(f"\nSummary for {checkpoint_label}")
    print("-" * 108)
    print(
        f"{'Split':<10} {'R@1':>8} {'R@5':>8} {'nDCG@10':>10} {'MRR':>8} {'TripAcc':>10} {'Margin':>10} {'Select':>10}"
    )
    print("-" * 108)
    for split_name, split_result in checkpoint_result.items():
        retrieval = split_result["retrieval"]["aggregate"]
        triplets = split_result["triplets"]["aggregate"]
        selection = split_result["selection"]["score"]
        print(
            f"{split_name:<10} {retrieval['recall@1']:>8.4f} {retrieval['recall@5']:>8.4f} "
            f"{retrieval['ndcg@10']:>10.4f} {retrieval['mrr']:>8.4f} {triplets['accuracy']:>10.4f} "
            f"{triplets['margin']:>10.4f} {selection:>10.4f}"
        )
    print("-" * 108)


def parse_checkpoint_arg(value: str) -> tuple[str, Path]:
    """Parse a checkpoint specification of the form label=path."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Checkpoint must be provided as label=path"
        )
    label, path_str = value.split("=", 1)
    checkpoint_path = Path(path_str)
    if not label:
        raise argparse.ArgumentTypeError("Checkpoint label cannot be empty")
    return label, checkpoint_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate embedding checkpoints on retrieval benchmark")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory containing dev.jsonl and holdout.jsonl benchmark files.",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        required=True,
        help="Checkpoint spec in the form label=path. Repeat for multiple checkpoints.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev", "holdout"],
        choices=["dev", "holdout"],
        help="Benchmark splits to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file path for detailed results.",
    )
    parser.add_argument(
        "--max-records-per-split",
        type=int,
        default=None,
        help="Optional cap for fast smoke tests.",
    )
    parser.add_argument(
        "--audit-top-k",
        type=int,
        default=10,
        help="How many retrieved docs to keep in failure/top-k audit output.",
    )
    parser.add_argument(
        "--max-failure-examples",
        type=int,
        default=20,
        help="Maximum number of worst failure examples to persist per split.",
    )
    parser.add_argument(
        "--persist-topk-audit",
        action="store_true",
        help="Persist top-k retrieval audit rows for every pair query, not just misses.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    device = torch.device(args.device)
    checkpoints = [parse_checkpoint_arg(value) for value in args.checkpoints]

    benchmark_records: dict[str, list[dict[str, Any]]] = {}
    for split in args.splits:
        records = load_jsonl(args.benchmark_dir / f"{split}.jsonl")
        if args.max_records_per_split is not None:
            records = cap_records_balanced(records, args.max_records_per_split)
        benchmark_records[split] = records

    results: dict[str, Any] = {
        "benchmark_dir": str(args.benchmark_dir),
        "device": str(device),
        "selection_weights": SELECTION_WEIGHTS,
        "audit_top_k": int(args.audit_top_k),
        "max_failure_examples": int(args.max_failure_examples),
        "persist_topk_audit": bool(args.persist_topk_audit),
        "checkpoints": {},
    }

    for label, checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\n=== Evaluating {label}: {checkpoint_path} ===")
        tokenizer = load_tokenizer(checkpoint_path)
        model = ModernBertMultiTaskModel.load_checkpoint(str(checkpoint_path), device=str(device))
        model = model.to(device)
        model.eval()

        checkpoint_result: dict[str, Any] = {}
        for split_name, split_records in benchmark_records.items():
            split_result = evaluate_split(
                model=model,
                tokenizer=tokenizer,
                records=split_records,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                audit_top_k=max(1, int(args.audit_top_k)),
                max_failure_examples=max(0, int(args.max_failure_examples)),
                persist_topk_audit=bool(args.persist_topk_audit),
            )
            checkpoint_result[split_name] = split_result

            retrieval = split_result["retrieval"]["aggregate"]
            triplets = split_result["triplets"]["aggregate"]
            selection = split_result["selection"]["score"]
            print(
                f"  {split_name:<7} | R@1={retrieval['recall@1']:.4f} "
                f"R@5={retrieval['recall@5']:.4f} nDCG@10={retrieval['ndcg@10']:.4f} MRR={retrieval['mrr']:.4f} "
                f"| triplet_acc={triplets['accuracy']:.4f} margin={triplets['margin']:.4f} selection={selection:.4f}"
            )

        results["checkpoints"][label] = {
            "path": str(checkpoint_path),
            "results": checkpoint_result,
        }
        print_split_summary_table(label, checkpoint_result)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()

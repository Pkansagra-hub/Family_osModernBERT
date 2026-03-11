#!/usr/bin/env python
"""
MTEB Benchmark for FamilyOS UltraBERT v4.0.4

Runs standard MTEB STS (Semantic Textual Similarity) tasks to evaluate
embedding quality with the attentive pooling head.

Tasks:
  - STS-B (validation split from GLUE)
  - STS12, STS13, STS14, STS15, STS16
  - SICK-R (Sentences Involving Compositional Knowledge - Relatedness)

Usage:
    python scripts/run_mteb_benchmark.py
    python scripts/run_mteb_benchmark.py --tasks STSBenchmark SICK-R
    python scripts/run_mteb_benchmark.py --output results/mteb_v404.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_local_tokenizer(checkpoint_path: Path) -> Any:
    """Load a tokenizer from checkpoint with a fast-tokenizer fallback."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

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


# ---------------------------------------------------------------------------
# MTEB-compatible model wrapper
# ---------------------------------------------------------------------------
class UltraBERTMTEBWrapper:
    """Wraps FamilyOS UltraBERT Client as an MTEB-compatible encoder.

    MTEB expects a model with an `encode()` method that accepts a list of
    sentences and returns a numpy array of embeddings.
    """

    def __init__(
        self,
        model_path: str | None = None,
        backend: str = "pytorch",
        device: str = "auto",
        method: str = "embedding_head",
    ) -> None:
        import torch

        self.method = method
        self._torch = torch
        self.client = None
        self.engine = None
        self.model_path = model_path
        self.backend = backend

        if model_path:
            from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

            checkpoint_path = Path(model_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            resolved_device = device
            if resolved_device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = ModernBertMultiTaskModel.load_checkpoint(
                str(checkpoint_path),
                device=resolved_device,
            )
            self.model = self.model.to(resolved_device)
            self.model.eval()
            self.encoder = getattr(self.model, "encoder", None)
            self.tokenizer = load_local_tokenizer(checkpoint_path)
            self.runtime_device = resolved_device
        else:
            from familyos_ultrabert import Client

            self.client = Client(
                model_path=model_path,
                backend=backend,
                device=device,
                warmup=True,
                warmup_rounds=3,
                verbose=True,
            )

            ultrabert = self.client._model
            if ultrabert is None:
                raise RuntimeError("Client failed to initialize UltraBERT")

            if not hasattr(ultrabert, "_engine"):
                raise RuntimeError("UltraBERT backend does not expose a PyTorch engine")

            self.engine = ultrabert._engine
            self.model = getattr(self.engine, "model", None)
            self.encoder = getattr(self.engine, "encoder", None)
            self.tokenizer = getattr(self.engine, "tokenizer", None)
            self.runtime_device = getattr(self.engine, "device", device)

        if method != "embedding_head" and backend != "pytorch":
            raise ValueError(f"Method '{method}' requires the PyTorch backend")

        if method in {"embedding_head", "encoder_mean"}:
            if self.model is None or self.encoder is None or self.tokenizer is None:
                raise RuntimeError("PyTorch engine is missing model, encoder, or tokenizer")

    def describe(self) -> Dict[str, Any]:
        """Return metadata about the active embedding method."""
        description: Dict[str, Any] = {
            "method": self.method,
            "backend": self.client.backend if self.client is not None else self.backend,
            "device": self.runtime_device,
        }

        if self.method == "embedding_head" and self.model is not None:
            heads = getattr(self.model, "heads", None)
            if heads is not None and "embedding" in heads:
                head = heads["embedding"]
                description.update(
                    {
                        "pooling": getattr(head, "pooling", "unknown"),
                        "params": int(sum(param.numel() for param in head.parameters())),
                    }
                )
        elif self.method == "encoder_mean":
            description.update(
                {
                    "pooling": "mean",
                    "params": 0,
                }
            )

        return description

    def _encode_embedding_head(self, sentences: Sequence[str]) -> np.ndarray:
        """Encode with the trained embedding head in batches."""
        outputs: List[np.ndarray] = []
        self.model.eval()
        with self._torch.no_grad():
            for start in range(0, len(sentences), self.batch_size):
                batch = list(sentences[start : start + self.batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.runtime_device) for key, value in encoded.items()}
                model_out = self.model(
                    capability="embedding",
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_dict=True,
                )
                outputs.append(model_out.logits.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _encode_encoder_mean(self, sentences: Sequence[str]) -> np.ndarray:
        """Encode with raw encoder outputs and mean pooling."""
        outputs: List[np.ndarray] = []
        self.encoder.eval()
        with self._torch.no_grad():
            for start in range(0, len(sentences), self.batch_size):
                batch = list(sentences[start : start + self.batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.runtime_device) for key, value in encoded.items()}
                encoder_out = self.encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_dict=True,
                )
                hidden = encoder_out.last_hidden_state
                attention_mask = encoded["attention_mask"]
                mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=-1)
                outputs.append(pooled.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode sentences to embeddings (MTEB interface).

        Args:
            sentences: List of sentences to encode.
            batch_size: Batch size for encoding.

        Returns:
            numpy array of shape (n_sentences, embedding_dim).
        """
        self.batch_size = max(1, int(batch_size))

        if self.method == "embedding_head":
            return self._encode_embedding_head(sentences).astype(np.float32)
        if self.method == "encoder_mean":
            return self._encode_encoder_mean(sentences).astype(np.float32)

        raise ValueError(f"Unknown encoding method: {self.method}")


# ---------------------------------------------------------------------------
# STS evaluation (standalone, no mteb library needed)
# ---------------------------------------------------------------------------

# Dataset configs for HuggingFace
STS_TASKS: Dict[str, Dict[str, Any]] = {
    "STSBenchmark": {
        "dataset": "glue",
        "subset": "stsb",
        "split": "validation",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "label",
        "score_range": (0.0, 5.0),
    },
    "STS12": {
        "dataset": "mteb/sts12-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (0.0, 5.0),
    },
    "STS13": {
        "dataset": "mteb/sts13-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (0.0, 5.0),
    },
    "STS14": {
        "dataset": "mteb/sts14-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (0.0, 5.0),
    },
    "STS15": {
        "dataset": "mteb/sts15-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (0.0, 5.0),
    },
    "STS16": {
        "dataset": "mteb/sts16-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (0.0, 5.0),
    },
    "SICK-R": {
        "dataset": "mteb/sickr-sts",
        "subset": None,
        "split": "test",
        "sentence1_col": "sentence1",
        "sentence2_col": "sentence2",
        "score_col": "score",
        "score_range": (1.0, 5.0),
    },
}


def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity between two embedding matrices."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-9)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-9)
    return np.sum(a_norm * b_norm, axis=1)


def evaluate_sts_task(
    model: UltraBERTMTEBWrapper,
    task_name: str,
    config: Dict[str, Any],
    batch_size: int,
) -> Dict[str, Any]:
    """Evaluate a single STS task.

    Returns dict with spearman, pearson correlations and metadata.
    """
    from datasets import load_dataset

    print(f"\n  [{task_name}] Loading dataset...")
    if config.get("subset"):
        dataset = load_dataset(
            config["dataset"],
            config["subset"],
            split=config["split"],
        )
    else:
        dataset = load_dataset(config["dataset"], split=config["split"])

    sentences1 = dataset[config["sentence1_col"]]
    sentences2 = dataset[config["sentence2_col"]]
    gold_scores = np.array(dataset[config["score_col"]], dtype=np.float32)

    num_samples = len(sentences1)
    print(f"  [{task_name}] {num_samples} pairs, encoding...")

    start = time.time()
    emb1 = model.encode(sentences1, batch_size=batch_size)
    emb2 = model.encode(sentences2, batch_size=batch_size)
    encode_time = time.time() - start

    # Cosine similarities
    pred_scores = cosine_similarity_batch(emb1, emb2)

    # Correlations
    spearman_corr = float(spearmanr(gold_scores, pred_scores).statistic)
    pearson_corr = float(pearsonr(gold_scores, pred_scores).statistic)

    # Cosine similarity stats
    cos_mean = float(np.mean(pred_scores))
    cos_std = float(np.std(pred_scores))

    print(f"  [{task_name}] Spearman: {spearman_corr:.4f} | Pearson: {pearson_corr:.4f}")
    print(f"  [{task_name}] Cosine stats: mean={cos_mean:.4f}, std={cos_std:.4f}")
    print(f"  [{task_name}] Encoding time: {encode_time:.1f}s ({num_samples / encode_time:.0f} pairs/s)")

    return {
        "task": task_name,
        "num_samples": num_samples,
        "spearman": round(spearman_corr, 4),
        "pearson": round(pearson_corr, 4),
        "cosine_mean": round(cos_mean, 4),
        "cosine_std": round(cos_std, 4),
        "encode_time_sec": round(encode_time, 2),
        "throughput_pairs_per_sec": round(num_samples / encode_time, 1),
    }


def run_mteb_sts(
    tasks: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    backend: str = "pytorch",
    device: str = "auto",
    methods: Optional[List[str]] = None,
    batch_size: int = 64,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run MTEB STS evaluation.

    Args:
        tasks: List of task names to run. None = all tasks.
        backend: UltraBERT backend ("pytorch" or "onnx").
        device: Device for inference.
        methods: Embedding methods to compare.
        batch_size: Batch size for encoding.
        output_path: Optional path to save JSON results.

    Returns:
        Dictionary with all results and summary.
    """
    if tasks is None:
        tasks = list(STS_TASKS.keys())
    if methods is None:
        methods = ["embedding_head"]

    invalid = [t for t in tasks if t not in STS_TASKS]
    if invalid:
        print(f"Unknown tasks: {invalid}")
        print(f"Available: {list(STS_TASKS.keys())}")
        sys.exit(1)

    valid_methods = {"embedding_head", "encoder_mean"}
    invalid_methods = [m for m in methods if m not in valid_methods]
    if invalid_methods:
        print(f"Unknown methods: {invalid_methods}")
        print(f"Available: {sorted(valid_methods)}")
        sys.exit(1)

    print("=" * 70)
    print("MTEB STS Benchmark - FamilyOS UltraBERT")
    print("=" * 70)

    # Get package version
    from familyos_ultrabert import __version__
    print(f"Package version: {__version__}")
    print(f"Model path: {model_path or '[package default]'}")
    print(f"Backend: {backend}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Methods: {', '.join(methods)}")

    # Initialize models
    print("\nInitializing model(s)...")
    models: Dict[str, UltraBERTMTEBWrapper] = {}
    for method in methods:
        print(f"  Loading method: {method}")
        model = UltraBERTMTEBWrapper(model_path=model_path, backend=backend, device=device, method=method)
        models[method] = model
        description = model.describe()
        print(
            f"  {method}: pooling={description.get('pooling', 'n/a')}, "
            f"params={description.get('params', 'n/a')}, device={description.get('device', 'unknown')}"
        )

    # Run tasks
    print("\n" + "-" * 70)
    print("Running STS evaluation tasks")
    print("-" * 70)

    overall_start = time.time()
    method_results: Dict[str, List[Dict[str, Any]]] = {}

    for method_name, model in models.items():
        print(f"\n{'#' * 70}")
        print(f"Method: {method_name}")
        print(f"{'#' * 70}")
        task_results: List[Dict[str, Any]] = []
        for task_name in tasks:
            config = STS_TASKS[task_name]
            result = evaluate_sts_task(model, task_name, config, batch_size=batch_size)
            result["method"] = method_name
            task_results.append(result)
        method_results[method_name] = task_results

    overall_time = time.time() - overall_start

    # Summary
    summaries: Dict[str, Dict[str, Any]] = {}
    for method_name, task_results in method_results.items():
        spearman_scores = [r["spearman"] for r in task_results]
        pearson_scores = [r["pearson"] for r in task_results]
        avg_spearman = float(np.mean(spearman_scores))
        avg_pearson = float(np.mean(pearson_scores))
        summaries[method_name] = {
            "version": __version__,
            "backend": backend,
            "method": method_name,
            "num_tasks": len(task_results),
            "avg_spearman": round(avg_spearman, 4),
            "avg_pearson": round(avg_pearson, 4),
            "total_time_sec": round(sum(item["encode_time_sec"] for item in task_results), 2),
        }

    full_results = {
        "summary": {
            "version": __version__,
            "backend": backend,
            "methods": methods,
            "batch_size": batch_size,
            "overall_time_sec": round(overall_time, 2),
        },
        "method_summaries": summaries,
        "methods": method_results,
    }

    # Print summary table
    print("\n" + "=" * 92)
    print("RESULTS SUMMARY")
    print("=" * 92)
    print(
        f"{'Method':<16} {'Task':<18} {'Spearman':>10} {'Pearson':>10} {'Samples':>8} {'Time':>8}"
    )
    print("-" * 92)
    for method_name, task_results in method_results.items():
        for index, result in enumerate(task_results):
            label = method_name if index == 0 else ""
            print(
                f"{label:<16} {result['task']:<18} {result['spearman']:>10.4f} {result['pearson']:>10.4f} "
                f"{result['num_samples']:>8} {result['encode_time_sec']:>7.1f}s"
            )
        summary = summaries[method_name]
        print(
            f"{'':<16} {'AVERAGE':<18} {summary['avg_spearman']:>10.4f} {summary['avg_pearson']:>10.4f} "
            f"{'':>8} {summary['total_time_sec']:>7.1f}s"
        )
        print("-" * 92)
    print(f"Total benchmark time: {overall_time:.1f}s")
    print("=" * 92)

    # Save results
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return full_results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MTEB STS Benchmark for FamilyOS UltraBERT",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=list(STS_TASKS.keys()),
        help="Specific STS tasks to run (default: all)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["embedding_head"],
        choices=["embedding_head", "encoder_mean"],
        help="Embedding methods to run side-by-side",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a local checkpoint/model directory to benchmark",
    )
    parser.add_argument(
        "--backend",
        default="pytorch",
        choices=["pytorch", "onnx"],
        help="Model backend",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    run_mteb_sts(
        tasks=args.tasks,
        model_path=args.model_path,
        backend=args.backend,
        device=args.device,
        methods=args.methods,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

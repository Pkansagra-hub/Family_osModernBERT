#!/usr/bin/env python
"""Run reconciliation embedding PoC experiments against a Modeling_studio checkpoint.

This launcher bridges the external PoC at ``d:/familyos/poc/reconciliation_embedding_poc``
with checkpoints produced in this repository. It can:

1. point the PoC embedder at a local Modeling_studio checkpoint
2. force the PoC to use the repository loader for checkpoint compatibility
3. delete old embedding caches
4. rebuild the corpus embedding cache before experiments
5. run one PoC experiment module at a time for dossier comparison

Example:
    python scripts/poc/run_reconciliation_embedding_poc.py --experiment exp1 --clear-cache --rebuild-cache
    python scripts/poc/run_reconciliation_embedding_poc.py --experiment exp9
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_FAMILYOS_ROOT = Path(r"d:/familyos")
DEFAULT_POC_ROOT = DEFAULT_FAMILYOS_ROOT / "poc" / "reconciliation_embedding_poc"
DEFAULT_MODELING_STUDIO_SRC = Path(r"d:/Modeling_studio/src")
DEFAULT_CHECKPOINT = Path(r"d:/Modeling_studio/checkpoints/distil_stage_b_bestema")
DEFAULT_CORPUS_PATH = (
    DEFAULT_FAMILYOS_ROOT / "k0" / "deploy" / "scripts" / "events" / "life_events.jsonl"
)

DOSSIER_SECTIONS: dict[str, str] = {
    "exp1": "EXP-1: Raw Embedding Quality Baseline",
    "exp2": "EXP-2: Centroid vs First-Event vs Summary Embedding",
    "exp3": "EXP-3: Reconciliation Threshold Calibration",
    "exp3b": "EXP-3b: Golden Calibration on Human-Labeled Pairs",
    "exp3c": "EXP-3c: Multi-Feature Reconciliation Classifier",
    "exp4": "EXP-4: K1 Signal Bypass Validation",
    "exp5": "EXP-5: Temporal + Semantic 2D Threshold Surface",
    "exp6": "EXP-6: Narrative Thread Coherence at Shared Locations",
    "exp7": "EXP-7: Episode Size Effect on Centroid Quality",
    "exp8": "EXP-8: Multi-Actor Disambiguation",
    "exp9": "EXP-9: Final Integrated Reconciliation Run",
    "exp10": "EXP-10: Candidate-Thread Inference",
    "exp11": "EXP-11: Final Policy Matrix Search",
}


def infer_backend_for_checkpoint(checkpoint: Path, requested_backend: str) -> str:
    """Choose a safe backend for the supplied checkpoint.

    Raw training checkpoints contain files like ``model.safetensors`` and should
    be loaded through the PyTorch repo-loader path, not the ONNX runtime path.
    """
    normalized = requested_backend.strip().lower()
    if normalized != "auto":
        return normalized

    if (checkpoint / "model.safetensors").exists() or (checkpoint / "pytorch_model.bin").exists():
        return "pytorch"

    onnx_files = list(checkpoint.glob("*.onnx"))
    if onnx_files:
        return "onnx"

    return "pytorch"

EXPERIMENT_ALIASES: dict[str, str] = {
    "exp0": "exp0_hard_negatives",
    "exp1": "exp1_baseline",
    "exp2": "exp2_centroid",
    "exp3": "exp3_thresholds",
    "exp3b": "exp3b_golden_calibration",
    "exp3c": "exp3c_multifeature_classifier",
    "exp4": "exp4_k1_signals",
    "exp5": "exp5_temporal",
    "exp6": "exp6_narrative",
    "exp7": "exp7_episode_size",
    "exp8": "exp8_multi_actor",
    "exp9": "exp9_final_run",
    "exp10": "exp10_candidate_thread_inference",
    "exp11": "exp11_policy_matrix",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run reconciliation embedding PoC experiments with a local Modeling_studio checkpoint."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment alias or module stem, e.g. exp1, exp9, exp10_candidate_thread_inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint directory to evaluate",
    )
    parser.add_argument(
        "--familyos-root",
        type=Path,
        default=DEFAULT_FAMILYOS_ROOT,
        help="Root directory that contains the external poc package",
    )
    parser.add_argument(
        "--modeling-studio-src",
        type=Path,
        default=DEFAULT_MODELING_STUDIO_SRC,
        help="Path to Modeling_studio/src for repo-side checkpoint loading",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="Embedding backend passed through to familyos_ultrabert.Client",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed through to familyos_ultrabert.Client",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete all cached embedding npz files before running",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute and save the corpus embedding cache before the experiment",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Life events jsonl used for cache rebuilding",
    )
    parser.add_argument(
        "--skip-repo-loader",
        action="store_true",
        help="Do not force the repo-side checkpoint loader patch",
    )
    return parser.parse_args()


def resolve_experiment_module(experiment: str) -> tuple[str, str]:
    """Resolve a user alias to the concrete experiment module name."""
    normalized = experiment.strip().lower()
    if normalized in EXPERIMENT_ALIASES:
        module_stem = EXPERIMENT_ALIASES[normalized]
    else:
        module_stem = experiment.strip()

    if not module_stem.startswith("exp"):
        raise ValueError(f"Experiment must start with 'exp': {experiment}")

    prefix = module_stem.split("_", 1)[0].lower()
    module_name = f"poc.reconciliation_embedding_poc.experiments.{module_stem}"
    return module_name, prefix


def configure_environment(
    checkpoint: Path,
    backend: str,
    device: str,
    modeling_studio_src: Path,
    use_repo_loader: bool,
) -> None:
    """Configure environment variables consumed by the external PoC embedder."""
    os.environ["RECON_EMBED_MODEL_PATH"] = str(checkpoint)
    os.environ["RECON_EMBED_BACKEND"] = backend
    os.environ["RECON_EMBED_DEVICE"] = device
    os.environ["RECON_EMBED_REPO_SRC"] = str(modeling_studio_src)
    os.environ["RECON_EMBED_USE_REPO_LOADER"] = "1" if use_repo_loader else "0"


def ensure_python_paths(familyos_root: Path, modeling_studio_src: Path) -> None:
    """Expose the external PoC repo plus local Modeling_studio packages to Python."""
    modeling_studio_root = modeling_studio_src.parent
    for path in (familyos_root, modeling_studio_root, modeling_studio_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def clear_embedding_cache_files(data_dir: Path) -> list[Path]:
    """Delete all cached embedding npz files in the PoC data directory."""
    removed: list[Path] = []
    for cache_file in sorted(data_dir.glob("embedding_cache_*.npz")):
        cache_file.unlink()
        removed.append(cache_file)
    return removed


def load_corpus_texts(corpus_path: Path) -> list[str]:
    """Load body.text entries from the FamilyOS life events corpus."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

    texts: list[str] = []
    with open(corpus_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line.strip())
            text = (row.get("body", {}) or {}).get("text", "")
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return texts


def rebuild_embedding_cache(poc_root: Path, corpus_path: Path) -> dict[str, Any]:
    """Recompute the corpus embedding cache using the configured checkpoint."""
    sys.path.insert(0, str(poc_root.parent.parent))
    from poc.reconciliation_embedding_poc.utils.embedder import Embedder

    texts = load_corpus_texts(corpus_path)
    embedder = Embedder()
    started = time.perf_counter()
    embedder.embed_batch(texts)
    embedder.save_cache()
    elapsed = time.perf_counter() - started
    stats = embedder.stats()
    stats["n_corpus_texts"] = len(texts)
    stats["rebuild_seconds"] = round(elapsed, 2)
    return stats


def snapshot_results(results_dir: Path) -> dict[Path, float]:
    """Capture result file modification times for before/after diffing."""
    snapshot: dict[Path, float] = {}
    for path in results_dir.glob("*"):
        if path.is_file():
            snapshot[path] = path.stat().st_mtime
    return snapshot


def discover_new_results(before: dict[Path, float], after: dict[Path, float]) -> list[Path]:
    """Return new or updated result files."""
    changed: list[Path] = []
    for path, mtime in after.items():
        if path not in before or before[path] != mtime:
            changed.append(path)
    return sorted(changed)


def resolve_run_callable(module: Any) -> Callable[[], Any]:
    """Find the single experiment entrypoint in a module."""
    run_callables = []
    for name in dir(module):
        if name.startswith("run_exp") and callable(getattr(module, name)):
            run_callables.append(getattr(module, name))
    if len(run_callables) != 1:
        raise RuntimeError(
            f"Expected exactly one run_exp* callable in {module.__name__}, found {len(run_callables)}"
        )
    return run_callables[0]


def main() -> None:
    """Launch the requested experiment with the configured checkpoint."""
    args = parse_args()
    poc_root = args.familyos_root / "poc" / "reconciliation_embedding_poc"
    results_dir = poc_root / "results"
    data_dir = poc_root / "data"

    if not poc_root.exists():
        raise FileNotFoundError(f"PoC root not found: {poc_root}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.modeling_studio_src.exists():
        raise FileNotFoundError(f"Modeling_studio src not found: {args.modeling_studio_src}")

    module_name, dossier_key = resolve_experiment_module(args.experiment)
    dossier_section = DOSSIER_SECTIONS.get(dossier_key, "Unknown dossier section")
    resolved_backend = infer_backend_for_checkpoint(args.checkpoint, args.backend)

    configure_environment(
        checkpoint=args.checkpoint,
        backend=resolved_backend,
        device=args.device,
        modeling_studio_src=args.modeling_studio_src,
        use_repo_loader=not args.skip_repo_loader,
    )
    ensure_python_paths(args.familyos_root, args.modeling_studio_src)

    print("=" * 80)
    print("Reconciliation Embedding PoC Launcher")
    print("=" * 80)
    print(f"Experiment module : {module_name}")
    print(f"Dossier section   : {dossier_section}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"PoC root          : {poc_root}")
    print(f"Results dir       : {results_dir}")
    print(f"Cache dir         : {data_dir}")
    print(f"Backend/device    : {resolved_backend}/{args.device}")
    print(f"Repo loader       : {'enabled' if not args.skip_repo_loader else 'disabled'}")

    if args.clear_cache:
        removed = clear_embedding_cache_files(data_dir)
        print(f"Removed cache files: {len(removed)}")
        for path in removed:
            print(f"  - {path.name}")

    if args.rebuild_cache:
        print("\nRebuilding embedding cache from life_events corpus ...")
        cache_stats = rebuild_embedding_cache(poc_root, args.corpus_path)
        print(json.dumps(cache_stats, indent=2))

    before = snapshot_results(results_dir)

    module = importlib.import_module(module_name)
    run_callable = resolve_run_callable(module)

    print("\nRunning experiment ...")
    started = time.perf_counter()
    result = run_callable()
    elapsed = time.perf_counter() - started

    after = snapshot_results(results_dir)
    changed = discover_new_results(before, after)

    print("\nRun complete")
    print(f"Elapsed seconds   : {elapsed:.2f}")
    print(f"Changed results   : {len(changed)}")
    for path in changed:
        print(f"  - {path.name}")

    if isinstance(result, dict):
        print(f"Returned keys     : {sorted(result.keys())}")

    print("\nNext step: compare the changed result files with")
    print(f"  {args.familyos_root / 'poc' / 'reconciliation_embedding_poc' / 'research_dossier.md'}")
    print(f"section: {dossier_section}")


if __name__ == "__main__":
    main()

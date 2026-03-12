"""Benchmark runner.

Milestone 1 runner is intentionally lightweight:
- loads `familyos_ultrabert.Client`
- discovers suites from `familyos_ultrabert.benchmarks.suite`
- runs suites and aggregates results

Per-capability suites are introduced in later milestones.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from familyos_ultrabert import __version__
from familyos_ultrabert.benchmarks.types import (
    BenchmarkResult,
    BenchmarkSeverity,
    BenchmarkRunResult,
    BenchmarkStatus,
    BenchmarkSummary,
    SuiteResult,
)


def _best_effort_git_info() -> Dict[str, Any]:
    """Collect git provenance (best-effort).

    Returns:
        Dict with commit/branch/dirty state when available.
    """
    import subprocess

    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True)
            val = str(out).strip()
            return val if val else None
        except Exception:  # noqa: BLE001
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty),
    }


def _best_effort_dependency_versions() -> Dict[str, Any]:
    """Collect dependency versions (best-effort)."""
    versions: Dict[str, Any] = {}
    for name in ("numpy", "tokenizers", "transformers", "onnxruntime", "torch"):
        try:
            mod = __import__(name)  # noqa: WPS421
            versions[name] = getattr(mod, "__version__", None)
        except Exception:  # noqa: BLE001
            versions[name] = None
    return versions


def _best_effort_hardware_info(client: Any) -> Dict[str, Any]:
    """Collect basic hardware/runtime provenance (best-effort)."""
    import platform
    import sys

    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
    }

    # Torch GPU details (optional)
    try:
        import torch  # type: ignore

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if bool(torch.cuda.is_available()):
            try:
                info["gpu_name"] = str(torch.cuda.get_device_name(0))
            except Exception:  # noqa: BLE001
                info["gpu_name"] = None
            info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:  # noqa: BLE001
        pass

    # Active backend/device from client when available
    info["client_backend"] = getattr(client, "backend", None)
    info["client_device"] = getattr(client, "device", None)
    return info


def _best_effort_model_hash(client: Any) -> Dict[str, Any]:
    """Compute a SHA256 hash over the weights used by the active backend.

    For ONNX this hashes all *.onnx files under weights/onnx.
    For PyTorch this hashes model.safetensors plus key config/tokenizer files.
    """
    import hashlib
    from pathlib import Path

    try:
        from familyos_ultrabert.model import DEFAULT_ONNX_PATH, DEFAULT_PYTORCH_PATH
    except Exception:  # noqa: BLE001
        return {"sha256": None, "files": []}

    backend = str(getattr(client, "backend", "unknown"))
    client_model_path = getattr(client, "_model_path", None)
    files: List[Path] = []

    if client_model_path:
        root = Path(str(client_model_path))
        if backend == "onnx":
            files = sorted(root.glob("*.onnx"))
            if not files:
                files = sorted((root / "onnx").glob("*.onnx")) if (root / "onnx").exists() else []
        elif backend == "pytorch":
            candidates = [
                root / "model.safetensors",
                root / "pytorch_model.bin",
                root / "config.json",
                root / "tokenizer.json",
                root / "tokenizer_config.json",
                root / "embedding_metadata.json",
                root / "capabilities.json",
            ]
            files = [p for p in candidates if p.exists()]
    elif backend == "onnx":
        root = Path(str(DEFAULT_ONNX_PATH))
        if root.exists():
            files = sorted(root.glob("*.onnx"))
    elif backend == "pytorch":
        root = Path(str(DEFAULT_PYTORCH_PATH))
        candidates = [
            root / "model.safetensors",
            root / "config.json",
            root / "tokenizer.json",
            root / "tokenizer_config.json",
        ]
        files = [p for p in candidates if p.exists()]

    if not files:
        return {"sha256": None, "files": []}

    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        try:
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        except OSError:
            # If hashing fails, return partial provenance rather than crashing.
            return {"sha256": None, "files": [str(x) for x in files]}

    return {"sha256": h.hexdigest(), "files": [str(x) for x in files]}


class BenchmarkRunner:
    """Runs benchmark suites and returns structured results."""

    def __init__(
        self,
        *,
        suites: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        backend: str = "auto",
        device: str = "auto",
        warmup_rounds: int = 3,
        verbose: bool = True,
    ):
        self._suite_filter = suites
        self._model_path = model_path
        self._backend = backend
        self._device = device
        self._warmup_rounds = warmup_rounds
        self._verbose = verbose

    def _log(self, message: str) -> None:
        if self._verbose:
            print(message)

    def _create_client(self) -> Any:
        """Create a configured client instance."""
        from familyos_ultrabert import Client

        return Client(
            model_path=self._model_path,
            backend=self._backend,
            device=self._device,
            warmup=True,
            warmup_rounds=self._warmup_rounds,
            verbose=self._verbose,
        )

    def _discover_suites(self) -> List[Any]:
        """Discover suite classes."""
        from familyos_ultrabert.benchmarks.suite import get_suite_classes

        suite_classes = get_suite_classes()
        if self._suite_filter:
            suite_classes = [c for c in suite_classes if getattr(c, "name", "") in self._suite_filter]
        return suite_classes

    def run(self) -> BenchmarkRunResult:
        """Run all discovered suites.

        Returns:
            BenchmarkRunResult containing all suite results.
        """
        overall_start = time.time()

        suite_results: List[SuiteResult] = []
        suite_classes = self._discover_suites()

        # If no suites are registered, avoid loading the model.
        if not suite_classes:
            duration = time.time() - overall_start
            summary = BenchmarkSummary(
                total=0,
                passed=0,
                failed=0,
                warned=0,
                info=0,
                skipped=0,
                errored=0,
                duration_sec=duration,
            )
            return BenchmarkRunResult(
                version=__version__,
                backend="unknown",
                suites=[],
                summary=summary,
                metadata={
                    "note": "No benchmark suites are registered.",
                    "runner": {
                        "backend": self._backend,
                        "warmup_rounds": self._warmup_rounds,
                        "suite_filter": self._suite_filter,
                    },
                },
            )

        client = self._create_client()
        active_backend = getattr(client, "backend", "unknown")
        # Prefer the backend's own notion of device (Client/inference backends
        # typically store this as a simple string like 'cpu' or 'cuda').
        active_device = getattr(client, "device", None)
        if active_device is None:
            active_device = getattr(getattr(client, "_backend", None), "device", None)

        self._log(f"UltraBERT benchmark runner (package={__version__}, backend={active_backend})")

        for suite_cls in suite_classes:
            suite_name = getattr(suite_cls, "name", suite_cls.__name__)
            self._log("")
            self._log(f"Running suite: {suite_name}")

            suite_start = time.time()
            suite = suite_cls(client)
            try:
                results = suite.run()
            except Exception as exc:  # noqa: BLE001
                # Keep runner resilient: record error as a single result.
                results = [
                    BenchmarkResult(
                        name="suite_error",
                        category=suite_name,
                        status=BenchmarkStatus.ERROR,
                        severity=BenchmarkSeverity.FAIL,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                ]

            elapsed = time.time() - suite_start
            suite_results.append(SuiteResult(suite_name=suite_name, results=list(results), total_time_sec=elapsed))

        # Aggregate counts
        all_results: List[BenchmarkResult] = []
        for s in suite_results:
            all_results.extend(s.results)

        passed = sum(1 for r in all_results if r.status == BenchmarkStatus.PASS)
        failed = sum(
            1
            for r in all_results
            if r.status == BenchmarkStatus.FAIL and r.severity == BenchmarkSeverity.FAIL
        )
        warned = sum(
            1
            for r in all_results
            if r.status == BenchmarkStatus.FAIL and r.severity == BenchmarkSeverity.WARN
        )
        info = sum(1 for r in all_results if r.severity == BenchmarkSeverity.INFO)
        skipped = sum(1 for r in all_results if r.status == BenchmarkStatus.SKIP)
        errored = sum(1 for r in all_results if r.status == BenchmarkStatus.ERROR)

        duration = time.time() - overall_start
        summary = BenchmarkSummary(
            total=len(all_results),
            passed=passed,
            failed=failed,
            warned=warned,
            info=info,
            skipped=skipped,
            errored=errored,
            duration_sec=duration,
        )

        metadata: Dict[str, Any] = {
            "client_backend": active_backend,
            "device": str(active_device) if active_device is not None else "unknown",
            "client_stats": getattr(client, "stats", None),
            "runner": {
                "model_path": self._model_path,
                "backend": self._backend,
                "warmup_rounds": self._warmup_rounds,
                "suite_filter": self._suite_filter,
            },
        }

        metadata["git"] = _best_effort_git_info()
        metadata["dependencies"] = _best_effort_dependency_versions()
        metadata["hardware"] = _best_effort_hardware_info(client)
        metadata["model"] = _best_effort_model_hash(client)

        # Convert dataclass stats if necessary
        if metadata.get("client_stats") is not None and hasattr(metadata["client_stats"], "__dict__"):
            try:
                metadata["client_stats"] = asdict(metadata["client_stats"])  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                pass

        return BenchmarkRunResult(
            version=__version__,
            backend=active_backend,
            suites=suite_results,
            summary=summary,
            metadata=metadata,
        )

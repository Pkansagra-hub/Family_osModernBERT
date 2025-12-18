#!/usr/bin/env python3
"""
Unified Inference Runtime for familyos_ultrabert

Multi-backend ONNX runtime with automatic fallback chain.
Supports: AMD NPU (DirectML), NVIDIA CUDA, AMD ROCm, CPU

Backend Priority:
    1. DirectML (AMD NPU on Windows - for Ryzen AI laptops)
    2. CUDA (NVIDIA GPU)
    3. ROCm (AMD GPU on Linux)
    4. CPU (Universal fallback)

Features:
    - Automatic backend detection and selection
    - Silent fallback (never crashes)
    - Environment-aware (Docker, Windows, Linux)
    - Session caching for performance
    - Memory-efficient with explicit cleanup

Usage:
    from familyos_ultrabert.runtime import get_session, get_available_backends

    # Get available backends
    backends = get_available_backends()
    print(f"Available: {backends}")

    # Create optimized session (auto-selects best backend)
    session = get_session("model.onnx")

    # Or specify backend explicitly
    session = get_session("model.onnx", backend="directml")

    # Run inference
    outputs = session.run(None, {"input": data})
"""

from __future__ import annotations

import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Backend Detection
# =============================================================================


class Backend(Enum):
    """Supported inference backends."""

    DIRECTML = "directml"  # AMD NPU (Windows only)
    CUDA = "cuda"          # NVIDIA GPU
    ROCM = "rocm"          # AMD GPU (Linux)
    CPU = "cpu"            # Universal fallback

    def __str__(self) -> str:
        return self.value


# Provider name mapping for ONNX Runtime
BACKEND_TO_PROVIDER: Dict[Backend, str] = {
    Backend.DIRECTML: "DmlExecutionProvider",
    Backend.CUDA: "CUDAExecutionProvider",
    Backend.ROCM: "ROCMExecutionProvider",
    Backend.CPU: "CPUExecutionProvider",
}


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def is_linux() -> bool:
    """Check if running on Linux."""
    return sys.platform == "linux"


def is_docker() -> bool:
    """Check if running inside Docker container."""
    # Check for Docker-specific files
    if Path("/.dockerenv").exists():
        return True

    # Check cgroup
    cgroup_path = Path("/proc/1/cgroup")
    if cgroup_path.exists():
        content = cgroup_path.read_text()
        if "docker" in content or "kubepods" in content:
            return True

    return False


def check_directml_available() -> bool:
    """
    Check if DirectML is available.

    DirectML requires:
    - Windows OS
    - DirectX 12 compatible GPU
    - onnxruntime-directml package
    """
    if not is_windows():
        return False

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return "DmlExecutionProvider" in providers
    except ImportError:
        return False


def check_cuda_available() -> bool:
    """
    Check if CUDA is available.

    CUDA requires:
    - NVIDIA GPU
    - CUDA toolkit and cuDNN
    - onnxruntime-gpu package
    """
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return "CUDAExecutionProvider" in providers
    except ImportError:
        return False


def check_rocm_available() -> bool:
    """
    Check if ROCm is available.

    ROCm requires:
    - AMD GPU on Linux
    - ROCm toolkit
    - onnxruntime-rocm package
    """
    if not is_linux():
        return False

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return "ROCMExecutionProvider" in providers
    except ImportError:
        return False


def get_available_backends() -> List[Backend]:
    """
    Get list of available backends in priority order.

    Returns:
        List of available backends, sorted by priority
    """
    available = []

    # Priority: DirectML > CUDA > ROCm > CPU
    if check_directml_available():
        available.append(Backend.DIRECTML)

    if check_cuda_available():
        available.append(Backend.CUDA)

    if check_rocm_available():
        available.append(Backend.ROCM)

    # CPU is always available
    available.append(Backend.CPU)

    return available


def get_best_backend() -> Backend:
    """
    Get the best available backend.

    Returns:
        Best available backend for current environment
    """
    backends = get_available_backends()
    return backends[0]


def get_provider_priority() -> List[str]:
    """
    Get ONNX Runtime providers in priority order.

    This is designed for use with InferenceSession's providers parameter.

    Returns:
        List of provider names in priority order
    """
    backends = get_available_backends()
    return [BACKEND_TO_PROVIDER[b] for b in backends]


# =============================================================================
# Session Management
# =============================================================================


# Global session cache
_session_cache: Dict[str, "ONNXSession"] = {}


def get_session(
    model_path: Union[str, Path],
    backend: Optional[Union[Backend, str]] = None,
    use_cache: bool = True,
    session_options: Optional[dict] = None,
) -> "ONNXSession":
    """
    Get or create an ONNX Runtime session for a model.

    Args:
        model_path: Path to ONNX model file
        backend: Specific backend to use (None = auto-select best)
        use_cache: Cache session for reuse
        session_options: Additional session configuration options

    Returns:
        ONNXSession wrapper with the loaded model
    """
    model_path = Path(model_path)
    cache_key = str(model_path.absolute())

    if use_cache and cache_key in _session_cache:
        cached = _session_cache[cache_key]
        if cached.is_valid():
            return cached
        else:
            # Remove stale cache entry
            del _session_cache[cache_key]

    # Create new session
    session = ONNXSession(model_path, backend=backend, options=session_options)

    if use_cache:
        _session_cache[cache_key] = session

    return session


def clear_session_cache():
    """Clear all cached sessions and free memory."""
    global _session_cache
    _session_cache.clear()


class ONNXSession:
    """
    ONNX Runtime session wrapper with automatic backend fallback.

    Features:
    - Automatic backend selection with fallback chain
    - Silent error handling (logs but doesn't crash)
    - Memory management with explicit cleanup
    - Provider-specific optimizations
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        backend: Optional[Union[Backend, str]] = None,
        options: Optional[dict] = None,
    ):
        """
        Initialize ONNX session with automatic backend selection.

        Args:
            model_path: Path to ONNX model
            backend: Specific backend (None = auto-select)
            options: Session configuration options
        """
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self._session = None
        self._backend = None
        self._input_names = []
        self._output_names = []

        # Parse backend if string
        if isinstance(backend, str):
            backend = Backend(backend.lower())

        # Get providers to try
        if backend is not None:
            # Specific backend requested with CPU fallback
            providers = [BACKEND_TO_PROVIDER[backend], "CPUExecutionProvider"]
        else:
            # Auto-select with full fallback chain
            providers = get_provider_priority()

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Apply custom options
        if options:
            if "intra_op_num_threads" in options:
                sess_options.intra_op_num_threads = options["intra_op_num_threads"]
            if "inter_op_num_threads" in options:
                sess_options.inter_op_num_threads = options["inter_op_num_threads"]
            if "enable_mem_pattern" in options:
                sess_options.enable_mem_pattern = options["enable_mem_pattern"]

        # Try providers in order with fallback
        last_error = None
        for provider in providers:
            try:
                logger.debug(f"Trying provider: {provider}")
                self._session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_options,
                    providers=[provider],
                )

                # Verify provider is actually used
                active_providers = self._session.get_providers()
                if provider in active_providers:
                    self._backend = provider
                    logger.info(f"Loaded {self.model_path.name} with {provider}")
                    break
                else:
                    # Provider not active, try next
                    logger.debug(f"{provider} not active, falling back...")
                    continue

            except Exception as e:
                last_error = e
                logger.debug(f"{provider} failed: {e}")
                continue

        if self._session is None:
            raise RuntimeError(
                f"Failed to load model {self.model_path} with any provider. "
                f"Last error: {last_error}"
            )

        # Cache input/output info
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    def run(
        self,
        output_names: Optional[List[str]],
        input_feed: Dict[str, any],
    ) -> List:
        """
        Run inference on the model.

        Args:
            output_names: List of output names to retrieve (None = all)
            input_feed: Dictionary of input name -> numpy array

        Returns:
            List of output arrays
        """
        return self._session.run(output_names, input_feed)

    def is_valid(self) -> bool:
        """Check if session is still valid."""
        return self._session is not None

    @property
    def backend(self) -> str:
        """Get active backend provider name."""
        return self._backend

    @property
    def input_names(self) -> List[str]:
        """Get list of input names."""
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        """Get list of output names."""
        return self._output_names

    def get_input_info(self) -> List[dict]:
        """Get detailed information about model inputs."""
        inputs = self._session.get_inputs()
        return [
            {
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            }
            for inp in inputs
        ]

    def get_output_info(self) -> List[dict]:
        """Get detailed information about model outputs."""
        outputs = self._session.get_outputs()
        return [
            {
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            }
            for out in outputs
        ]

    def close(self):
        """Release session resources."""
        self._session = None
        self._backend = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Provider-Specific Configuration
# =============================================================================


def get_directml_options() -> dict:
    """Get optimized options for DirectML (AMD NPU)."""
    return {
        "enable_mem_pattern": False,  # DirectML manages memory
    }


def get_cuda_options(device_id: int = 0) -> Tuple[str, dict]:
    """
    Get optimized configuration for CUDA.

    Args:
        device_id: CUDA device ID (0-based)

    Returns:
        Tuple of (provider_name, provider_options)
    """
    return (
        "CUDAExecutionProvider",
        {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        },
    )


def get_cpu_options(num_threads: Optional[int] = None) -> dict:
    """
    Get optimized options for CPU inference.

    Args:
        num_threads: Number of threads (None = auto)

    Returns:
        Session options dictionary
    """
    if num_threads is None:
        # Use half of available cores
        num_threads = max(1, os.cpu_count() // 2)

    return {
        "intra_op_num_threads": num_threads,
        "inter_op_num_threads": 2,
    }


# =============================================================================
# Diagnostics
# =============================================================================


def print_runtime_info():
    """Print runtime environment information."""
    print("=" * 60)
    print("familyos_ultrabert Runtime Information")
    print("=" * 60)

    print(f"\nPlatform: {sys.platform}")
    print(f"Python: {sys.version}")
    print(f"Docker: {is_docker()}")

    try:
        import onnxruntime as ort
        print(f"\nONNX Runtime: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
    except ImportError:
        print("\nONNX Runtime: NOT INSTALLED")

    print(f"\nAvailable backends: {[str(b) for b in get_available_backends()]}")
    print(f"Best backend: {get_best_backend()}")

    print("\nBackend details:")
    print(f"  DirectML: {'Available' if check_directml_available() else 'Not available'}")
    if not is_windows():
        print("    (DirectML requires Windows)")

    print(f"  CUDA: {'Available' if check_cuda_available() else 'Not available'}")
    print(f"  ROCm: {'Available' if check_rocm_available() else 'Not available'}")
    if not is_linux():
        print("    (ROCm requires Linux)")

    print(f"  CPU: Always available")

    print("=" * 60)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    # Run diagnostics when executed directly
    logging.basicConfig(level=logging.INFO)
    print_runtime_info()

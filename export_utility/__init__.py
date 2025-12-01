"""
Export Utility Module

Production-ready model export and optimization utilities for FamilyOS.

Modules:
    - export_model: HuggingFace/safetensors export
    - export_onnx: ONNX export with quantization
    - export_tensorrt: TensorRT optimization
    - prune_model: Model pruning utilities
    - benchmark_latency: Latency benchmarks
    - batch_optimizer: Dynamic batching for high throughput
    - optimized_inference: Single forward pass with parallel heads
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = [
    "export_model",
    "export_onnx",
    "export_tensorrt",
    "prune_model",
    "benchmark_latency",
    "batch_optimizer",
    "optimized_inference",
]

EXPORT_UTILITY_DIR = Path(__file__).parent

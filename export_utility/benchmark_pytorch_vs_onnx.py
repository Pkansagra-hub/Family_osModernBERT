#!/usr/bin/env python
"""
Benchmark: PyTorch OptimizedMultiTaskModel vs ONNX models

Compares latency for running multiple capabilities:
1. OptimizedMultiTaskModel (single encoder pass, parallel heads)
2. ONNX models (separate model per capability)

Usage:
    python export_utility/benchmark_pytorch_vs_onnx.py \
        --pytorch-model exports/pruned-15pct \
        --onnx-dir exports/onnx-pruned-15pct \
        --capabilities sentiment emotions safety_familyos intent ingress embedding
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_pytorch(model_path: str, capabilities: list[str], texts: list[str], num_runs: int = 50):
    """Benchmark PyTorch OptimizedMultiTaskModel."""
    from export_utility.optimized_inference import OptimizedMultiTaskModel

    print("\n" + "=" * 60)
    print("PYTORCH OptimizedMultiTaskModel")
    print("=" * 60)

    # Load model
    print(f"Loading model from {model_path}...")
    model = OptimizedMultiTaskModel.from_pretrained(
        model_path,
        device="cuda",
        enable_caching=False,  # Disable cache for fair comparison
        parallel_heads=True,
    )

    # Filter to available capabilities
    available_caps = [c for c in capabilities if c in model.capabilities]
    print(f"Capabilities: {available_caps}")

    # Warmup
    print("Warming up (5 runs)...")
    for text in texts[:5]:
        model.infer(text, available_caps, use_cache=False)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs per text)...")
    latencies = []

    for text in texts:
        for _ in range(num_runs):
            start = time.perf_counter()
            result = model.infer(text, available_caps, use_cache=False)
            latencies.append((time.perf_counter() - start) * 1000)

    results = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "num_capabilities": len(available_caps),
        "per_capability_ms": float(np.mean(latencies)) / len(available_caps),
    }

    print(f"\nResults ({len(available_caps)} capabilities, single pass):")
    print(f"  Mean:  {results['mean_ms']:.2f} ms")
    print(f"  P50:   {results['p50_ms']:.2f} ms")
    print(f"  P90:   {results['p90_ms']:.2f} ms")
    print(f"  P99:   {results['p99_ms']:.2f} ms")
    print(f"  Per capability: {results['per_capability_ms']:.2f} ms")

    return results


def benchmark_onnx_cpu(onnx_dir: str, capabilities: list[str], texts: list[str], num_runs: int = 50):
    """Benchmark ONNX models on CPU."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    print("\n" + "=" * 60)
    print("ONNX Runtime (CPU)")
    print("=" * 60)

    onnx_path = Path(onnx_dir)

    # Find available ONNX models
    available_caps = []
    models = {}

    for cap in capabilities:
        # Try quantized first, then base
        quantized = onnx_path / f"{cap}_quantized.onnx"
        base = onnx_path / f"{cap}.onnx"

        if quantized.exists():
            models[cap] = str(quantized)
            available_caps.append(cap)
        elif base.exists():
            models[cap] = str(base)
            available_caps.append(cap)

    print(f"Found ONNX models: {available_caps}")

    if not available_caps:
        print("No ONNX models found!")
        return None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(onnx_path.parent / "pruned-15pct")

    # Load ONNX sessions
    sessions = {}
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    for cap in available_caps:
        sessions[cap] = ort.InferenceSession(
            models[cap],
            sess_options,
            providers=["CPUExecutionProvider"],
        )

    # Warmup
    print("Warming up (5 runs)...")
    for text in texts[:5]:
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        for cap in available_caps:
            sessions[cap].run(None, dict(inputs))

    # Benchmark - running all capabilities sequentially (ONNX has no shared encoder)
    print(f"Benchmarking ({num_runs} runs per text)...")
    latencies = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        ort_inputs = dict(inputs)

        for _ in range(num_runs):
            start = time.perf_counter()
            # Run ALL capabilities (sequential, each has its own encoder)
            for cap in available_caps:
                sessions[cap].run(None, ort_inputs)
            latencies.append((time.perf_counter() - start) * 1000)

    results = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "num_capabilities": len(available_caps),
        "per_capability_ms": float(np.mean(latencies)) / len(available_caps),
    }

    print(f"\nResults ({len(available_caps)} capabilities, sequential):")
    print(f"  Mean:  {results['mean_ms']:.2f} ms")
    print(f"  P50:   {results['p50_ms']:.2f} ms")
    print(f"  P90:   {results['p90_ms']:.2f} ms")
    print(f"  P99:   {results['p99_ms']:.2f} ms")
    print(f"  Per capability: {results['per_capability_ms']:.2f} ms")

    return results


def benchmark_onnx_gpu(onnx_dir: str, capabilities: list[str], texts: list[str], num_runs: int = 50):
    """Benchmark ONNX models on GPU."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    # Check if CUDA provider is available
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        print("\nCUDAExecutionProvider not available, skipping GPU benchmark")
        return None

    print("\n" + "=" * 60)
    print("ONNX Runtime (GPU - CUDA)")
    print("=" * 60)

    onnx_path = Path(onnx_dir)

    # Find available ONNX models (use base models for GPU, not quantized)
    available_caps = []
    models = {}

    for cap in capabilities:
        base = onnx_path / f"{cap}.onnx"
        if base.exists():
            models[cap] = str(base)
            available_caps.append(cap)

    print(f"Found ONNX models: {available_caps}")

    if not available_caps:
        print("No ONNX models found!")
        return None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(onnx_path.parent / "pruned-15pct")

    # Load ONNX sessions with CUDA
    sessions = {}
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    for cap in available_caps:
        sessions[cap] = ort.InferenceSession(
            models[cap],
            sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    # Warmup
    print("Warming up (5 runs)...")
    for text in texts[:5]:
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        for cap in available_caps:
            sessions[cap].run(None, dict(inputs))

    # Benchmark
    print(f"Benchmarking ({num_runs} runs per text)...")
    latencies = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        ort_inputs = dict(inputs)

        for _ in range(num_runs):
            start = time.perf_counter()
            for cap in available_caps:
                sessions[cap].run(None, ort_inputs)
            latencies.append((time.perf_counter() - start) * 1000)

    results = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "num_capabilities": len(available_caps),
        "per_capability_ms": float(np.mean(latencies)) / len(available_caps),
    }

    print(f"\nResults ({len(available_caps)} capabilities, sequential):")
    print(f"  Mean:  {results['mean_ms']:.2f} ms")
    print(f"  P50:   {results['p50_ms']:.2f} ms")
    print(f"  P90:   {results['p90_ms']:.2f} ms")
    print(f"  P99:   {results['p99_ms']:.2f} ms")
    print(f"  Per capability: {results['per_capability_ms']:.2f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX inference")
    parser.add_argument("--pytorch-model", required=True, help="Path to PyTorch model")
    parser.add_argument("--onnx-dir", required=True, help="Path to ONNX models directory")
    parser.add_argument(
        "--capabilities",
        nargs="+",
        default=["sentiment", "emotions", "safety_familyos", "intent", "ingress", "embedding"],
        help="Capabilities to benchmark",
    )
    parser.add_argument("--num-runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--output", "-o", help="Output JSON file")

    args = parser.parse_args()

    # Test texts
    texts = [
        "Mom picked up Panda from school today.",
        "I hate you so much, you're the worst person ever!",
        "Can you help me with my homework please?",
        "The weather is nice today, let's go to the park.",
        "I'm feeling really sad and lonely right now.",
    ]

    print("=" * 60)
    print("BENCHMARK: PyTorch vs ONNX Multi-Capability Inference")
    print("=" * 60)
    print(f"Capabilities: {args.capabilities}")
    print(f"Test texts: {len(texts)}")
    print(f"Runs per text: {args.num_runs}")

    results = {}

    # Benchmark PyTorch
    results["pytorch_gpu"] = benchmark_pytorch(
        args.pytorch_model, args.capabilities, texts, args.num_runs
    )

    # Benchmark ONNX CPU
    results["onnx_cpu"] = benchmark_onnx_cpu(
        args.onnx_dir, args.capabilities, texts, args.num_runs
    )

    # Benchmark ONNX GPU
    results["onnx_gpu"] = benchmark_onnx_gpu(
        args.onnx_dir, args.capabilities, texts, args.num_runs
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Method':<25} {'Mean (ms)':<12} {'P50 (ms)':<12} {'Per-Cap (ms)':<12}")
    print("-" * 60)

    for method, res in results.items():
        if res:
            print(
                f"{method:<25} {res['mean_ms']:<12.2f} {res['p50_ms']:<12.2f} {res['per_capability_ms']:<12.2f}"
            )

    # Calculate speedups
    if results.get("pytorch_gpu") and results.get("onnx_cpu"):
        speedup = results["onnx_cpu"]["mean_ms"] / results["pytorch_gpu"]["mean_ms"]
        print(f"\nPyTorch GPU vs ONNX CPU: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    if results.get("pytorch_gpu") and results.get("onnx_gpu"):
        speedup = results["onnx_gpu"]["mean_ms"] / results["pytorch_gpu"]["mean_ms"]
        print(f"PyTorch GPU vs ONNX GPU: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    print("\nNote: PyTorch uses SINGLE encoder pass for all capabilities")
    print("      ONNX runs each capability SEPARATELY (N encoder passes)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

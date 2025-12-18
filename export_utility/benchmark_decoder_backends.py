#!/usr/bin/env python3
"""
Decoder Backend Benchmark: CUDA vs CPU vs NPU (DirectML)

Compares latency and generation quality across:
- CUDA (FP32) - NVIDIA GPU
- CUDA (INT8) - NVIDIA GPU with quantized model
- DirectML (FP32) - AMD NPU
- DirectML (INT8) - AMD NPU with quantized model
- CPU (FP32) - Baseline
- CPU (INT8) - CPU with quantized model

Usage:
    python export_utility/benchmark_decoder_backends.py \
        --onnx-dir exports/decoder-onnx-v3 \
        --encoder-checkpoint outputs/modernbert-v2-for-v3-transfer/checkpoint-18000 \
        --decoder-checkpoint outputs/ultrabert-gen-decoder-v3 \
        --num-runs 5 \
        --max-tokens 50

Output:
    - Latency comparison table
    - Tokens/second comparison
    - Generation quality samples
    - Memory usage (if available)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    backend: str
    quantization: str  # "fp32" or "int8"
    model_size_mb: float
    warmup_time_s: float
    latencies_ms: List[float] = field(default_factory=list)
    tokens_generated: List[int] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def mean_latency_ms(self) -> float:
        return np.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def std_latency_ms(self) -> float:
        return np.std(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def mean_tokens_per_second(self) -> float:
        if not self.latencies_ms or not self.tokens_generated:
            return 0.0
        tps_values = [
            (tokens / (latency / 1000.0))
            for tokens, latency in zip(self.tokens_generated, self.latencies_ms)
            if latency > 0
        ]
        return np.mean(tps_values) if tps_values else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return np.percentile(self.latencies_ms, 50) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return np.percentile(self.latencies_ms, 95) if self.latencies_ms else 0.0


def get_available_backends() -> List[Tuple[str, str]]:
    """Get list of available (backend, provider) pairs."""
    import onnxruntime as ort

    available = []
    providers = ort.get_available_providers()

    # Check each backend in priority order
    backend_map = [
        ("TensorRT", "TensorrtExecutionProvider"),
        ("CUDA", "CUDAExecutionProvider"),
        ("VitisAI (NPU)", "VitisAIExecutionProvider"),  # AMD Ryzen AI NPU
        ("DirectML", "DmlExecutionProvider"),  # DirectML runs on GPU, not NPU
        ("ROCm", "ROCMExecutionProvider"),
        ("CPU", "CPUExecutionProvider"),
    ]

    for name, provider in backend_map:
        if provider in providers:
            available.append((name, provider))

    return available


def get_model_size_mb(path: Path) -> float:
    """Get model file size in MB."""
    if not path.exists():
        return 0.0

    total = path.stat().st_size

    # Check for external data file
    data_file = Path(str(path) + ".data")
    if data_file.exists():
        total += data_file.stat().st_size

    return total / (1024 * 1024)


def create_session(
    model_path: Path,
    provider: str,
) -> Optional["ort.InferenceSession"]:
    """Create ONNX Runtime session with specified provider."""
    import onnxruntime as ort

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Provider-specific options
        if provider == "CUDAExecutionProvider":
            provider_options = [
                (provider, {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                }),
            ]
        elif provider == "TensorrtExecutionProvider":
            provider_options = [
                (provider, {
                    "device_id": 0,
                    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2GB
                    "trt_fp16_enable": True,
                }),
                ("CUDAExecutionProvider", {"device_id": 0}),  # Fallback
            ]
        elif provider == "DmlExecutionProvider":
            provider_options = [
                (provider, {}),
            ]
        elif provider == "VitisAIExecutionProvider":
            # AMD Ryzen AI NPU - VitisAI execution provider
            provider_options = [
                (provider, {
                    "config_file": "",  # Use default config
                }),
            ]
        else:
            provider_options = [provider]

        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=provider_options if provider not in ["CPUExecutionProvider"] else [provider],
        )

        return session
    except Exception as e:
        print(f"Failed to create session with {provider}: {e}")
        return None


def warmup_session(
    prefix_session: "ort.InferenceSession",
    decoder_session: "ort.InferenceSession",
    prefix_shape: Tuple[int, int, int],
) -> float:
    """Warmup sessions and return warmup time."""
    start = time.perf_counter()

    # Warmup prefix encoder
    dummy_hidden = np.random.randn(*prefix_shape).astype(np.float32)
    prefix_session.run(None, {"encoder_hidden_states": dummy_hidden})

    # Warmup decoder with minimal input
    dummy_prefix = np.random.randn(1, prefix_shape[1], 1024).astype(np.float32)
    dummy_ids = np.array([[50281]], dtype=np.int64)  # BOS token
    dummy_mask = np.ones((1, prefix_shape[1] + 1), dtype=np.float32)

    decoder_session.run(None, {
        "prefix_embeds": dummy_prefix,
        "decoder_input_ids": dummy_ids,
        "attention_mask": dummy_mask,
    })

    return time.perf_counter() - start


def generate_tokens(
    prefix_session: "ort.InferenceSession",
    decoder_session: "ort.InferenceSession",
    encoder_hidden: np.ndarray,
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> Tuple[List[int], float]:
    """Generate tokens and return (token_ids, latency_ms)."""
    BOS, EOS = 50281, 50282

    start = time.perf_counter()

    # Project encoder hidden states
    prefix_embeds = prefix_session.run(
        None,
        {"encoder_hidden_states": encoder_hidden}
    )[0]
    prefix_len = prefix_embeds.shape[1]

    # Generate tokens
    generated_ids = [BOS]

    for step in range(max_tokens):
        dec_ids = np.array([generated_ids], dtype=np.int64)
        attn_mask = np.ones((1, prefix_len + len(generated_ids)), dtype=np.float32)

        logits = decoder_session.run(None, {
            "prefix_embeds": prefix_embeds,
            "decoder_input_ids": dec_ids,
            "attention_mask": attn_mask,
        })[0]

        next_logits = logits[0, -1, :].copy()

        # Repetition penalty
        for prev_token in set(generated_ids):
            if next_logits[prev_token] > 0:
                next_logits[prev_token] /= repetition_penalty
            else:
                next_logits[prev_token] *= repetition_penalty

        # Temperature
        next_logits = next_logits / temperature

        # Top-p sampling
        sorted_indices = np.argsort(next_logits)[::-1]
        sorted_logits = next_logits[sorted_indices]
        probs = np.exp(sorted_logits - np.max(sorted_logits))
        probs = probs / probs.sum()
        cumsum = np.cumsum(probs)
        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = probs[:cutoff_idx]
        top_probs = top_probs / top_probs.sum()

        # Sample
        next_token = int(np.random.choice(top_indices, p=top_probs))
        generated_ids.append(next_token)

        if next_token == EOS:
            break

    latency_ms = (time.perf_counter() - start) * 1000

    return generated_ids[1:], latency_ms  # Exclude BOS


def run_benchmark(
    onnx_dir: Path,
    encoder_checkpoint: Path,
    decoder_checkpoint: Path,
    test_inputs: List[str],
    num_runs: int,
    max_tokens: int,
) -> List[BenchmarkResult]:
    """Run benchmark across all available backends and quantization levels."""
    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModel
    from safetensors.torch import load_file

    print("=" * 80)
    print("DECODER BACKEND BENCHMARK")
    print("=" * 80)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(decoder_checkpoint))
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load encoder (PyTorch - needed for encoding input)
    print("\nLoading encoder...")
    config = AutoConfig.from_pretrained(str(encoder_checkpoint))
    encoder = AutoModel.from_config(config)
    weights = load_file(str(encoder_checkpoint / "model.safetensors"))
    encoder_weights = {k[8:]: v for k, v in weights.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_weights, strict=False)
    encoder.eval()
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")

    # Pre-encode all test inputs
    print("\nEncoding test inputs...")
    encoded_inputs = []
    for text in test_inputs:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = encoder(**inputs)
            hidden = outputs.last_hidden_state.numpy().astype(np.float32)
        encoded_inputs.append(hidden)
        print(f"  Input shape: {hidden.shape}")

    # Get available backends
    print("\nDetecting available backends...")
    available = get_available_backends()
    print(f"Available backends: {[b[0] for b in available]}")

    # Model variants to test
    model_variants = [
        ("fp32", "prefix_encoder.onnx", "decoder.onnx"),
        ("int8", "prefix_encoder_int8.onnx", "decoder_int8.onnx"),
    ]

    results: List[BenchmarkResult] = []

    # Run benchmarks for each backend and quantization
    for backend_name, provider in available:
        for quant, prefix_file, decoder_file in model_variants:
            prefix_path = onnx_dir / prefix_file
            decoder_path = onnx_dir / decoder_file

            # Skip if model files don't exist
            if not prefix_path.exists() or not decoder_path.exists():
                print(f"\nSkipping {backend_name} {quant}: model files not found")
                continue

            print(f"\n{'=' * 60}")
            print(f"Benchmark: {backend_name} ({quant.upper()})")
            print("=" * 60)

            # Calculate model size
            prefix_size = get_model_size_mb(prefix_path)
            decoder_size = get_model_size_mb(decoder_path)
            total_size = prefix_size + decoder_size
            print(f"Model size: {total_size:.2f} MB (prefix: {prefix_size:.2f}, decoder: {decoder_size:.2f})")

            result = BenchmarkResult(
                backend=backend_name,
                quantization=quant,
                model_size_mb=total_size,
                warmup_time_s=0.0,
            )

            # Create sessions
            print(f"Creating sessions with {provider}...")
            prefix_sess = create_session(prefix_path, provider)
            decoder_sess = create_session(decoder_path, provider)

            if prefix_sess is None or decoder_sess is None:
                result.errors.append(f"Failed to create session with {provider}")
                results.append(result)
                continue

            # Warmup
            print("Warming up...")
            try:
                warmup_time = warmup_session(
                    prefix_sess,
                    decoder_sess,
                    encoded_inputs[0].shape,
                )
                result.warmup_time_s = warmup_time
                print(f"Warmup time: {warmup_time:.3f}s")
            except Exception as e:
                result.errors.append(f"Warmup failed: {e}")
                results.append(result)
                continue

            # Run benchmark
            print(f"Running {num_runs} iterations...")
            for run_idx in range(num_runs):
                for enc_hidden in encoded_inputs:
                    try:
                        tokens, latency = generate_tokens(
                            prefix_sess,
                            decoder_sess,
                            enc_hidden,
                            max_tokens=max_tokens,
                        )

                        result.latencies_ms.append(latency)
                        result.tokens_generated.append(len(tokens))

                        output_text = tokenizer.decode(tokens, skip_special_tokens=True)
                        result.outputs.append(output_text)

                        # Print first output for quality check
                        if run_idx == 0 and len(result.outputs) == 1:
                            print(f"Sample output: {output_text[:100]}...")

                    except Exception as e:
                        result.errors.append(f"Run {run_idx}: {str(e)}")

                print(f"  Run {run_idx + 1}/{num_runs}: "
                      f"mean={result.mean_latency_ms:.1f}ms, "
                      f"{result.mean_tokens_per_second:.1f} tokens/s")

            results.append(result)

            # Cleanup
            del prefix_sess, decoder_sess
            gc.collect()

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    # Header
    headers = [
        "Backend",
        "Quant",
        "Size (MB)",
        "Mean (ms)",
        "Std (ms)",
        "P50 (ms)",
        "P95 (ms)",
        "Tokens/s",
        "Speedup",
    ]

    print(f"\n{'Backend':<20} {'Quant':<6} {'Size':>10} {'Mean':>10} {'Std':>8} "
          f"{'P50':>10} {'P95':>10} {'Tok/s':>10} {'Speedup':>8}")
    print("-" * 100)

    # Find CPU FP32 as baseline
    cpu_fp32_tps = None
    for r in results:
        if r.backend == "CPU" and r.quantization == "fp32" and not r.errors:
            cpu_fp32_tps = r.mean_tokens_per_second
            break

    for r in results:
        if r.errors:
            status = f"ERROR: {r.errors[0][:40]}"
            print(f"{r.backend:<20} {r.quantization:<6} {status}")
        else:
            speedup = (r.mean_tokens_per_second / cpu_fp32_tps) if cpu_fp32_tps else 1.0
            print(f"{r.backend:<20} {r.quantization:<6} {r.model_size_mb:>10.1f} "
                  f"{r.mean_latency_ms:>10.1f} {r.std_latency_ms:>8.1f} "
                  f"{r.p50_latency_ms:>10.1f} {r.p95_latency_ms:>10.1f} "
                  f"{r.mean_tokens_per_second:>10.1f} {speedup:>8.2f}x")

    print("-" * 100)

    # Best configuration
    best = max(
        [r for r in results if not r.errors],
        key=lambda r: r.mean_tokens_per_second,
        default=None,
    )

    if best:
        print(f"\nBest configuration: {best.backend} ({best.quantization.upper()})")
        print(f"  - {best.mean_tokens_per_second:.1f} tokens/second")
        print(f"  - {best.mean_latency_ms:.1f} ms mean latency")
        print(f"  - {best.model_size_mb:.1f} MB model size")


def save_results(results: List[BenchmarkResult], output_path: Path):
    """Save results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {
                "backend": r.backend,
                "quantization": r.quantization,
                "model_size_mb": r.model_size_mb,
                "warmup_time_s": r.warmup_time_s,
                "mean_latency_ms": r.mean_latency_ms,
                "std_latency_ms": r.std_latency_ms,
                "p50_latency_ms": r.p50_latency_ms,
                "p95_latency_ms": r.p95_latency_ms,
                "mean_tokens_per_second": r.mean_tokens_per_second,
                "num_runs": len(r.latencies_ms),
                "sample_output": r.outputs[0] if r.outputs else None,
                "errors": r.errors,
            }
            for r in results
        ]
    }

    output_path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark decoder across backends and quantization levels"
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("exports/decoder-onnx-v3"),
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=Path("outputs/modernbert-v2-for-v3-transfer/checkpoint-18000"),
        help="Path to encoder checkpoint",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=Path("outputs/ultrabert-gen-decoder-v3"),
        help="Path to decoder checkpoint (for tokenizer)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark runs per configuration",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmark_runs/decoder_backend_benchmark.json"),
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # Test inputs
    test_inputs = [
        "I have been feeling very stressed about work and cannot relax.",
        "My child keeps having nightmares and I don't know how to help them.",
        "We had an argument about finances and now we're not speaking.",
    ]

    # Run benchmark
    results = run_benchmark(
        onnx_dir=args.onnx_dir,
        encoder_checkpoint=args.encoder_checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
        test_inputs=test_inputs,
        num_runs=args.num_runs,
        max_tokens=args.max_tokens,
    )

    # Print results
    print_results_table(results)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, args.output)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

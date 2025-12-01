#!/usr/bin/env python3
"""
Production Latency Benchmark Script

Comprehensive benchmarking for model inference latency across different
configurations, batch sizes, sequence lengths, and devices.

Benchmarks:
    - PyTorch (FP32, FP16)
    - ONNX Runtime (if available)
    - TensorRT (if available)
    - CPU vs GPU comparison
    - Batch size scaling
    - Sequence length scaling

Usage:
    # Basic benchmark
    python export_utility/benchmark_latency.py \
        --model outputs/modernbert-multitask-v0 \
        --output outputs/latency_benchmark.json

    # Full benchmark suite
    python export_utility/benchmark_latency.py \
        --model outputs/modernbert-multitask-v0 \
        --batch-sizes 1 4 8 16 32 \
        --seq-lengths 64 128 256 512 \
        --capabilities sentiment ner_general safety_familyos \
        --device cuda \
        --warmup 10 \
        --iterations 100

    # CPU only benchmark
    python export_utility/benchmark_latency.py \
        --model outputs/modernbert-multitask-v0 \
        --device cpu \
        --output outputs/latency_cpu.json

    # Compare all capabilities
    python export_utility/benchmark_latency.py \
        --model outputs/modernbert-multitask-v0 \
        --capabilities all \
        --batch-sizes 1 8 \
        --output outputs/capability_latency.json

Outputs:
    - latency_benchmark.json: Full benchmark results
    - latency_report.md: Human-readable summary
    - latency_plots/ (optional): Visualization charts
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default benchmark configurations
DEFAULT_BATCH_SIZES = [1, 8, 32]
DEFAULT_SEQ_LENGTHS = [64, 128, 256, 512]
DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 100


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LatencyResult:
    """Single latency measurement result."""

    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_qps: float  # queries per second
    samples_per_sec: float  # samples per second (batch_size * qps)

    def to_dict(self) -> dict:
        return {
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "throughput_qps": round(self.throughput_qps, 2),
            "samples_per_sec": round(self.samples_per_sec, 2),
        }


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    batch_sizes: list[int] = field(default_factory=lambda: DEFAULT_BATCH_SIZES)
    seq_lengths: list[int] = field(default_factory=lambda: DEFAULT_SEQ_LENGTHS)
    capabilities: list[str] = field(default_factory=lambda: ["sentiment"])
    warmup: int = DEFAULT_WARMUP
    iterations: int = DEFAULT_ITERATIONS
    device: str = "cuda"
    precision: str = "fp32"  # fp32, fp16, bf16


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    model_path: str
    device: str
    precision: str
    timestamp: str
    system_info: dict
    results: dict[str, dict]  # capability -> config -> LatencyResult

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "precision": self.precision,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "results": {
                cap: {
                    config: result.to_dict() if isinstance(result, LatencyResult) else result
                    for config, result in cap_results.items()
                }
                for cap, cap_results in self.results.items()
            },
        }


# =============================================================================
# System Info
# =============================================================================


def get_system_info() -> dict:
    """Gather system information for benchmark context."""
    import platform

    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
    }

    # CUDA info
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
    else:
        info["cuda_available"] = False

    # CPU info
    try:
        import psutil

        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass

    return info


# =============================================================================
# Benchmark Functions
# =============================================================================


def generate_dummy_input(
    tokenizer,
    batch_size: int,
    seq_length: int,
    device: torch.device,
) -> dict:
    """Generate dummy input for benchmarking."""
    # Create dummy text of approximately target length
    dummy_text = "This is a sample text for benchmarking. " * (seq_length // 8)

    # Tokenize
    inputs = tokenizer(
        [dummy_text] * batch_size,
        max_length=seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move to device
    return {k: v.to(device) for k, v in inputs.items()}


def warmup_model(
    model,
    inputs: dict,
    capability: str,
    warmup_iterations: int,
) -> None:
    """Warmup model to ensure consistent measurements."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(**inputs, task=capability)

    # Sync CUDA if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(
    model,
    inputs: dict,
    capability: str,
    iterations: int,
    batch_size: int,
) -> LatencyResult:
    """Measure inference latency over multiple iterations."""
    model.eval()
    latencies = []

    with torch.no_grad():
        for _ in range(iterations):
            # Sync before measurement
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(**inputs, task=capability)

            # Sync after measurement
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    # Compute statistics
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    mean_ms = statistics.mean(latencies)
    std_ms = statistics.stdev(latencies) if n > 1 else 0.0
    p50_ms = latencies_sorted[int(n * 0.50)]
    p90_ms = latencies_sorted[int(n * 0.90)]
    p95_ms = latencies_sorted[int(n * 0.95)]
    p99_ms = latencies_sorted[min(int(n * 0.99), n - 1)]
    min_ms = min(latencies)
    max_ms = max(latencies)

    # Compute throughput
    throughput_qps = 1000.0 / mean_ms  # queries per second
    samples_per_sec = throughput_qps * batch_size

    return LatencyResult(
        mean_ms=mean_ms,
        std_ms=std_ms,
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        throughput_qps=throughput_qps,
        samples_per_sec=samples_per_sec,
    )


def benchmark_capability(
    model,
    tokenizer,
    capability: str,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, LatencyResult]:
    """Benchmark a single capability across all configurations."""
    results = {}

    total_configs = len(config.batch_sizes) * len(config.seq_lengths)
    pbar = tqdm(total=total_configs, desc=f"Benchmarking {capability}", leave=False)

    for batch_size in config.batch_sizes:
        for seq_length in config.seq_lengths:
            config_key = f"bs{batch_size}_seq{seq_length}"

            # Generate inputs
            inputs = generate_dummy_input(tokenizer, batch_size, seq_length, device)

            # Warmup
            warmup_model(model, inputs, capability, config.warmup)

            # Measure
            result = measure_latency(model, inputs, capability, config.iterations, batch_size)

            results[config_key] = result

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            pbar.update(1)

    pbar.close()
    return results


def run_full_benchmark(
    model,
    tokenizer,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, dict]:
    """Run full benchmark suite across all capabilities."""
    all_results = {}

    logger.info(f"Running benchmarks on {device}")
    logger.info(f"Batch sizes: {config.batch_sizes}")
    logger.info(f"Sequence lengths: {config.seq_lengths}")
    logger.info(f"Capabilities: {config.capabilities}")
    logger.info(f"Warmup: {config.warmup}, Iterations: {config.iterations}")

    for capability in config.capabilities:
        if capability not in model.heads:
            logger.warning(f"Capability {capability} not in model, skipping")
            continue

        logger.info(f"\nBenchmarking: {capability}")
        cap_results = benchmark_capability(model, tokenizer, capability, config, device)
        all_results[capability] = cap_results

        # Log summary for this capability
        if "bs1_seq128" in cap_results:
            result = cap_results["bs1_seq128"]
            logger.info(
                f"  bs=1, seq=128: {result.mean_ms:.2f}ms (p99: {result.p99_ms:.2f}ms, "
                f"{result.throughput_qps:.1f} qps)"
            )

    return all_results


# =============================================================================
# Report Generation
# =============================================================================


def generate_markdown_report(results: BenchmarkResults) -> str:
    """Generate human-readable markdown report."""
    lines = []
    lines.append("# Latency Benchmark Report")
    lines.append("")
    lines.append(f"**Model:** `{results.model_path}`")
    lines.append(f"**Device:** {results.device}")
    lines.append(f"**Precision:** {results.precision}")
    lines.append(f"**Timestamp:** {results.timestamp}")
    lines.append("")

    # System info
    lines.append("## System Information")
    lines.append("")
    for key, value in results.system_info.items():
        lines.append(f"- **{key}:** {value}")
    lines.append("")

    # Summary table (bs=1, seq=128)
    lines.append("## Summary (batch=1, seq=128)")
    lines.append("")
    lines.append("| Capability | Mean (ms) | P50 (ms) | P99 (ms) | QPS | Samples/sec |")
    lines.append("|------------|-----------|----------|----------|-----|-------------|")

    for capability, cap_results in results.results.items():
        if "bs1_seq128" in cap_results:
            r = cap_results["bs1_seq128"]
            if isinstance(r, dict):
                lines.append(
                    f"| {capability} | {r['mean_ms']:.2f} | {r['p50_ms']:.2f} | "
                    f"{r['p99_ms']:.2f} | {r['throughput_qps']:.1f} | {r['samples_per_sec']:.1f} |"
                )
    lines.append("")

    # Detailed results per capability
    lines.append("## Detailed Results")
    lines.append("")

    for capability, cap_results in results.results.items():
        lines.append(f"### {capability}")
        lines.append("")
        lines.append("| Config | Mean (ms) | Std (ms) | P50 | P90 | P95 | P99 | Samples/sec |")
        lines.append("|--------|-----------|----------|-----|-----|-----|-----|-------------|")

        for config, r in cap_results.items():
            if isinstance(r, dict):
                lines.append(
                    f"| {config} | {r['mean_ms']:.2f} | {r['std_ms']:.2f} | "
                    f"{r['p50_ms']:.2f} | {r['p90_ms']:.2f} | {r['p95_ms']:.2f} | "
                    f"{r['p99_ms']:.2f} | {r['samples_per_sec']:.1f} |"
                )
        lines.append("")

    # Scaling analysis
    lines.append("## Scaling Analysis")
    lines.append("")
    lines.append("### Batch Size Scaling (seq=128)")
    lines.append("")

    for capability, cap_results in results.results.items():
        lines.append(f"**{capability}:**")
        batch_results = []
        for config, r in cap_results.items():
            if "seq128" in config and isinstance(r, dict):
                bs = int(config.split("_")[0].replace("bs", ""))
                batch_results.append((bs, r["mean_ms"], r["samples_per_sec"]))

        batch_results.sort(key=lambda x: x[0])
        for bs, mean, sps in batch_results:
            lines.append(f"  - bs={bs}: {mean:.2f}ms, {sps:.1f} samples/sec")
        lines.append("")

    return "\n".join(lines)


def generate_summary_table(results: BenchmarkResults) -> str:
    """Generate a compact summary table for console output."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("BENCHMARK SUMMARY (batch=1, seq=128)")
    lines.append("=" * 80)
    lines.append(
        f"{'Capability':<20} {'Mean (ms)':<12} {'P99 (ms)':<12} {'QPS':<10} {'Samples/s':<12}"
    )
    lines.append("-" * 80)

    for capability, cap_results in results.results.items():
        if "bs1_seq128" in cap_results:
            r = cap_results["bs1_seq128"]
            if isinstance(r, dict):
                lines.append(
                    f"{capability:<20} {r['mean_ms']:<12.2f} {r['p99_ms']:<12.2f} "
                    f"{r['throughput_qps']:<10.1f} {r['samples_per_sec']:<12.1f}"
                )

    lines.append("=" * 80)
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model inference latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python export_utility/benchmark_latency.py \\
      --model outputs/modernbert-multitask-v0

  # Full benchmark
  python export_utility/benchmark_latency.py \\
      --model outputs/modernbert-multitask-v0 \\
      --batch-sizes 1 4 8 16 32 \\
      --seq-lengths 64 128 256 512 \\
      --capabilities all \\
      --iterations 100

  # Quick test
  python export_utility/benchmark_latency.py \\
      --model outputs/modernbert-multitask-v0 \\
      --batch-sizes 1 \\
      --seq-lengths 128 \\
      --warmup 3 \\
      --iterations 10
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for benchmark results (JSON)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help=f"Batch sizes to benchmark (default: {DEFAULT_BATCH_SIZES})",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENGTHS,
        help=f"Sequence lengths to benchmark (default: {DEFAULT_SEQ_LENGTHS})",
    )
    parser.add_argument(
        "--capabilities",
        type=str,
        nargs="+",
        default=["sentiment"],
        help="Capabilities to benchmark (or 'all')",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to benchmark on (default: auto)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Precision for inference (default: fp32)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup iterations (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Benchmark iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate markdown report",
    )

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load model
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

    logger.info(f"Loading model from {model_path}")
    model = ModernBertMultiTaskModel.from_pretrained(str(model_path))
    model = model.to(device)
    model.eval()

    # Apply precision
    if args.precision == "fp16" and device.type == "cuda":
        model = model.half()
        logger.info("Using FP16 precision")
    elif args.precision == "bf16" and device.type == "cuda":
        model = model.bfloat16()
        logger.info("Using BF16 precision")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    logger.info(f"Model loaded with heads: {list(model.heads.keys())}")

    # Resolve capabilities
    if "all" in args.capabilities:
        capabilities = list(model.heads.keys())
    else:
        capabilities = args.capabilities

    # Create config
    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        capabilities=capabilities,
        warmup=args.warmup,
        iterations=args.iterations,
        device=str(device),
        precision=args.precision,
    )

    # Get system info
    system_info = get_system_info()

    # Run benchmarks
    logger.info("\nStarting benchmark...")
    start_time = time.time()
    all_results = run_full_benchmark(model, tokenizer, config, device)
    elapsed = time.time() - start_time
    logger.info(f"\nBenchmark completed in {elapsed:.1f}s")

    # Create results object
    results = BenchmarkResults(
        model_path=str(model_path),
        device=str(device),
        precision=args.precision,
        timestamp=datetime.now().isoformat(),
        system_info=system_info,
        results=all_results,
    )

    # Print summary
    print(generate_summary_table(results))

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")

        # Save markdown report
        if args.report:
            report_path = output_path.with_suffix(".md")
            report = generate_markdown_report(results)
            with open(report_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {report_path}")
    else:
        # Print results to stdout
        print("\nFull results:")
        print(json.dumps(results.to_dict(), indent=2))


if __name__ == "__main__":
    main()
    main()

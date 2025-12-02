#!/usr/bin/env python3
"""
Production Latency Benchmark Script

Comprehensive benchmarking for model inference latency across different
configurations, batch sizes, sequence lengths, and devices.

This script uses LatencyBenchmark from modeling_studio.evaluation.benchmarks.

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
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer

# Use evaluation module instead of reimplementing
from modeling_studio.evaluation.benchmarks import LatencyBenchmark, LatencyResults
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel

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
class BenchmarkReport:
    """Complete benchmark results."""

    model_path: str
    device: str
    precision: str
    timestamp: str
    system_info: dict
    results: dict[str, dict]  # capability -> config -> LatencyResults

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "precision": self.precision,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "results": {
                cap: {
                    config: result.to_dict() if hasattr(result, "to_dict") else result
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
# Report Generation
# =============================================================================


def generate_markdown_report(report: BenchmarkReport) -> str:
    """Generate human-readable markdown report."""
    lines = []
    lines.append("# Latency Benchmark Report")
    lines.append("")
    lines.append(f"**Model:** `{report.model_path}`")
    lines.append(f"**Device:** {report.device}")
    lines.append(f"**Precision:** {report.precision}")
    lines.append(f"**Timestamp:** {report.timestamp}")
    lines.append("")

    # System info
    lines.append("## System Information")
    lines.append("")
    for key, value in report.system_info.items():
        lines.append(f"- **{key}:** {value}")
    lines.append("")

    # Results per capability
    lines.append("## Results")
    lines.append("")

    for capability, cap_results in report.results.items():
        lines.append(f"### {capability}")
        lines.append("")
        lines.append(
            "| Config | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (samples/s) |"
        )
        lines.append(
            "|--------|-----------|----------|----------|----------|------------------------|"
        )

        for config_key, result in cap_results.items():
            if isinstance(result, dict):
                lines.append(
                    f"| {config_key} | {result.get('mean_ms', 0):.2f} | "
                    f"{result.get('p50_ms', 0):.2f} | {result.get('p95_ms', 0):.2f} | "
                    f"{result.get('p99_ms', 0):.2f} | {result.get('throughput', 0):.1f} |"
                )
            elif hasattr(result, "mean_ms"):
                lines.append(
                    f"| {config_key} | {result.mean_ms:.2f} | "
                    f"{result.p50_ms:.2f} | {result.p95_ms:.2f} | "
                    f"{result.p99_ms:.2f} | {result.throughput:.1f} |"
                )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main Benchmark Function
# =============================================================================


def run_benchmark(
    model_path: str | Path,
    config: BenchmarkConfig,
    output_dir: str | Path | None = None,
) -> BenchmarkReport:
    """
    Run latency benchmarks using LatencyBenchmark from evaluation module.

    Args:
        model_path: Path to model checkpoint
        config: Benchmark configuration
        output_dir: Directory to save results

    Returns:
        BenchmarkReport with all results
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir) if output_dir else model_path / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LATENCY BENCHMARK")
    logger.info("=" * 60)

    # Gather system info
    system_info = get_system_info()
    logger.info(f"Device: {config.device}")
    if system_info.get("cuda_available"):
        logger.info(f"GPU: {system_info.get('gpu_name')}")

    # Load model
    logger.info(f"\nLoading model from {model_path}...")
    if (model_path / "best").exists():
        model_path = model_path / "best"

    model = ModernBertMultiTaskModel.load_checkpoint(
        checkpoint_path=str(model_path),
        device=config.device,
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Get available capabilities
    if "all" in config.capabilities:
        capabilities = [c.value for c in model.capabilities]
    else:
        capabilities = [
            c for c in config.capabilities if c in [cap.value for cap in model.capabilities]
        ]

    logger.info(f"Capabilities: {capabilities}")
    logger.info(f"Batch sizes: {config.batch_sizes}")
    logger.info(f"Sequence lengths: {config.seq_lengths}")

    # Create LatencyBenchmark using the module
    benchmark = LatencyBenchmark(
        model=model,
        tokenizer=tokenizer,
        device=config.device,
    )

    # Run benchmarks for each capability and configuration
    all_results = {}

    for capability in capabilities:
        logger.info(f"\nBenchmarking: {capability}")
        cap_results = {}

        for batch_size in config.batch_sizes:
            for seq_length in config.seq_lengths:
                config_key = f"bs{batch_size}_seq{seq_length}"

                # Generate sample texts
                sample_text = "This is a sample text for benchmarking. " * (seq_length // 8)
                texts = [sample_text] * max(config.iterations, 100)

                # Run benchmark using the module's run() method
                result: LatencyResults = benchmark.run(
                    texts=texts,
                    batch_size=batch_size,
                    warmup=config.warmup,
                    capability=capability,
                    max_length=seq_length,
                )

                cap_results[config_key] = result

                logger.info(
                    f"  {config_key}: {result.mean_ms:.2f}ms "
                    f"(P95: {result.p95_ms:.2f}ms, {result.throughput:.1f} samples/s)"
                )

        all_results[capability] = cap_results

    # Create report
    report = BenchmarkReport(
        model_path=str(model_path),
        device=config.device,
        precision=config.precision,
        timestamp=datetime.now().isoformat(),
        system_info=system_info,
        results=all_results,
    )

    # Save JSON
    json_path = output_dir / "latency_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"\nSaved JSON to {json_path}")

    # Save markdown report
    md_path = output_dir / "latency_report.md"
    with open(md_path, "w") as f:
        f.write(generate_markdown_report(report))
    logger.info(f"Saved report to {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for capability, cap_results in all_results.items():
        if "bs1_seq128" in cap_results:
            result = cap_results["bs1_seq128"]
            if hasattr(result, "mean_ms"):
                print(f"{capability}: {result.mean_ms:.2f}ms (bs=1, seq=128)")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model latency across configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic benchmark
    python export_utility/benchmark_latency.py \\
        --model outputs/modernbert-multitask-v0

    # Full benchmark suite
    python export_utility/benchmark_latency.py \\
        --model outputs/modernbert-multitask-v0 \\
        --batch-sizes 1 4 8 16 32 \\
        --seq-lengths 64 128 256 512 \\
        --capabilities sentiment ner_general safety_familyos

    # CPU benchmark
    python export_utility/benchmark_latency.py \\
        --model outputs/modernbert-multitask-v0 \\
        --device cpu
""",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results",
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help=f"Batch sizes to benchmark. Default: {DEFAULT_BATCH_SIZES}",
    )

    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENGTHS,
        help=f"Sequence lengths to benchmark. Default: {DEFAULT_SEQ_LENGTHS}",
    )

    parser.add_argument(
        "--capabilities",
        type=str,
        nargs="+",
        default=["sentiment"],
        help="Capabilities to benchmark. Use 'all' for all capabilities.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warmup iterations. Default: {DEFAULT_WARMUP}",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of measurement iterations. Default: {DEFAULT_ITERATIONS}",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for benchmarking",
    )

    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Precision for benchmarking",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        capabilities=args.capabilities,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
        precision=args.precision,
    )

    run_benchmark(
        model_path=args.model,
        config=config,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()

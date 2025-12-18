#!/usr/bin/env python3
"""
Quantize ONNX models for AMD Ryzen AI NPU using AMD Quark.

This creates NPU-compatible INT8 models that run on VitisAIExecutionProvider.

Usage:
    python export_utility/quantize_for_npu.py \
        --input exports/decoder-onnx-v3/decoder.onnx \
        --output exports/decoder-onnx-v3/decoder_npu.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader


class RandomCalibrationDataReader(CalibrationDataReader):
    """Generate random calibration data for quantization."""

    def __init__(self, model_inputs, num_samples: int = 100):
        self.inputs = model_inputs
        self.num_samples = num_samples
        self.current = 0

    def get_next(self):
        if self.current >= self.num_samples:
            return None

        self.current += 1
        data = {}
        for inp in self.inputs:
            # Get shape from input
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)  # Dynamic dims default to 1

            # Get element type
            elem_type = inp.type.tensor_type.elem_type
            # ONNX elem types: 1=float, 7=int64, 6=int32, 9=bool
            if elem_type == 7:  # INT64
                data[inp.name] = np.random.randint(0, 1000, size=shape).astype(np.int64)
            elif elem_type == 6:  # INT32
                data[inp.name] = np.random.randint(0, 1000, size=shape).astype(np.int32)
            elif elem_type == 9:  # BOOL
                data[inp.name] = np.ones(shape, dtype=bool)
            else:  # Default to float32
                data[inp.name] = np.random.randn(*shape).astype(np.float32)

        return data

    def rewind(self):
        self.current = 0


def quantize_for_npu(
    input_path: Path,
    output_path: Path,
    calibration_samples: int = 100,
) -> None:
    """Quantize ONNX model for AMD NPU using Quark."""
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config.config import (
        Config,
        QuantizationConfig,
        CalibrationMethod,
    )
    from onnxruntime.quantization import QuantType, QuantFormat

    print(f"Loading model: {input_path}")

    # Load model to get input info
    import onnx
    model = onnx.load(str(input_path))

    # Get input info
    inputs = model.graph.input
    print(f"Model inputs: {[i.name for i in inputs]}")

    # Create quantization config for NPU (VitisAI)
    quant_config = QuantizationConfig(
        quant_format=QuantFormat.QDQ,  # Quantize-Dequantize format for NPU
        calibrate_method=CalibrationMethod.MinMax,  # MinMax calibration
        activation_type=QuantType.QInt8,  # INT8 activations
        weight_type=QuantType.QInt8,  # INT8 weights
        enable_npu_cnn=True,  # Enable NPU CNN optimizations
        enable_npu_transformer=True,  # Enable NPU transformer optimizations
        include_cle=True,  # Cross-layer equalization
        include_sq=False,  # Smooth quantization
        optimize_model=True,
        execution_providers=["CPUExecutionProvider"],  # Use CPU for calibration
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "EnableSubgraph": True,
            "MatMulConstBOnly": True,
        }
    )

    config = Config(global_quant_config=quant_config)

    # Create calibration data reader
    calibration_reader = RandomCalibrationDataReader(inputs, calibration_samples)

    print(f"Quantizing for NPU with {calibration_samples} calibration samples...")

    # Run quantization
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(
        str(input_path),
        str(output_path),
        calibration_reader,
    )

    print(f"NPU-optimized model saved to: {output_path}")

    # Print size comparison
    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Size: {input_size:.1f} MB -> {output_size:.1f} MB ({output_size/input_size*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model for AMD NPU")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("exports/decoder-onnx-v3/prefix_encoder.onnx"),
        help="Input ONNX model path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: input_npu.onnx)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_npu.onnx"

    quantize_for_npu(args.input, args.output, args.calibration_samples)


if __name__ == "__main__":
    main()
